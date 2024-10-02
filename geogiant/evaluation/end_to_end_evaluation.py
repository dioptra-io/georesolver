import asyncio

from tqdm import tqdm
from random import shuffle
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pych_client import ClickHouseClient
from ipaddress import IPv4Address, AddressValueError


from geogiant.clickhouse import CreateDNSMappingTable, GetDstPrefix, InsertFromCSV
from geogiant.prober import RIPEAtlasAPI, RIPEAtlasProber
from geogiant.ecs_mapping_init import resolve_vps_subnet
from geogiant.evaluation.plot import plot_router_2ms
from geogiant.evaluation.scores import get_scores
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.queries import (
    load_vps,
    get_pings_per_target,
    get_subnets,
)
from geogiant.common.files_utils import (
    load_csv,
    dump_csv,
    load_json,
    dump_json,
    load_json_iter,
    load_anycatch_data,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# INPUT TABLE
PING_RIPE_IP_MAP_TABLE = "pings_ripe_ip_map"

# OUTPUT TABLES
PING_END_TO_END_TABLE = "pings_end_to_end"
ECS_TABLE = "end_to_end_mapping_ecs"
VPS_ECS_MAPPING_TABLE = "vps_mapping_ecs"

# FILE PATHS
RIPE_IP_MAP_SUBNETS = path_settings.END_TO_END_DATASET / "ripe_ip_map_subnets.json"
ROUTERS_SUBNETS = path_settings.END_TO_END_DATASET / "routers_subnets.json"
END_TO_END_HOSTNAMES_PATH = (
    path_settings.END_TO_END_DATASET / "end_to_end_hostnames.csv"
)
filtered_end_to_end_targets_path = (
    path_settings.END_TO_END_DATASET / "filtered_end_to_end_targets.json"
)
end_to_end_targets_path = path_settings.END_TO_END_DATASET / "end_to_end_targets.json"
END_TO_END_SUBNETS_PATH = path_settings.END_TO_END_DATASET / "end_to_end_subnets.json"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"
SCORE_FILE = (
    path_settings.RESULTS_PATH
    / f"end_to_end_evaluation/scores__best_hostname_geo_score.pickle"
)
RESULT_FILE = (
    path_settings.RESULTS_PATH
    / f"end_to_end_evaluation/{'results' + str(SCORE_FILE).split('scores')[-1]}"
)

# CONSTANT PARAMETERS
probing_budget = 50


def generate_routers_targets_file() -> None:
    if not (end_to_end_targets_path).exists():
        rows = load_json_iter(path_settings.END_TO_END_DATASET / "routers_2ms.json")
        targets = {}
        for row in rows:
            addr = row["ip"]
            # remove IPv6 and private IP addresses
            try:
                if not IPv4Address(addr).is_private:
                    subnet = get_prefix_from_ip(addr)
                    targets[addr] = {
                        "subnet": subnet,
                        "lat": row["probe_latitude"],
                        "lon": row["probe_longitude"],
                    }
            except AddressValueError:
                continue

        logger.info(f"Number of targets in routers datasets:: {len(targets)}")

        dump_json(targets, end_to_end_targets_path)


def load_targets(target_subnets: list) -> None:

    if not filtered_end_to_end_targets_path.exists():
        filtered_targets = []

        generate_routers_targets_file()
        all_targets = load_json(end_to_end_targets_path)
        for target_addr, target_info in all_targets.items():
            if not target_info["subnet"] in target_subnets:
                continue

            filtered_targets.append(target_addr)

        dump_json(filtered_targets, filtered_end_to_end_targets_path)

    filtered_targets = load_json(filtered_end_to_end_targets_path)

    return filtered_targets


def load_ripe_ip_map_subnets(max_ripe_ip_map_subnet: int = 1_500) -> list:
    """retrieve all RIPE IP map subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetDstPrefix().execute(
            client=client,
            table_name=PING_RIPE_IP_MAP_TABLE,
        )

    ripe_ip_map_subnets = []
    for row in rows:
        ripe_ip_map_subnets.append(row["dst_prefix"])

    shuffle(ripe_ip_map_subnets)

    return ripe_ip_map_subnets[:max_ripe_ip_map_subnet]


def load_routers_2ms_subnets(max_subnets_routers: int = 5_000) -> list[str]:
    routers_2ms_subnets = load_json(
        path_settings.END_TO_END_DATASET / "routers_2ms_subnets.json"
    )
    routers_ip_addresses = load_csv(
        path_settings.END_TO_END_DATASET / "2500_random_routers_more_than_2ms.csv"
    )
    routers_subnets = list(set([get_prefix_from_ip(ip) for ip in routers_ip_addresses]))

    shuffle(routers_2ms_subnets)
    routers_2ms_subnets = routers_2ms_subnets[: len(routers_subnets)]

    routers_subnets.extend(routers_2ms_subnets)

    return routers_subnets


def load_hostnames() -> list[str]:
    selected_hostnames = set()

    pair_config = [(10, 3), (20, 3), (50, 3)]
    for bgp_threshold, nb_hostname_per_ns_org in pair_config:
        selected_hostnames_per_cdn_per_ns = load_json(
            path_settings.DATASET
            / f"hostname_geo_score_selection_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.json"
        )

        hostnames_per_config = set()
        for ns in selected_hostnames_per_cdn_per_ns:
            for _, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
                selected_hostnames.update(hostnames)
                hostnames_per_config.update(hostnames)

        logger.debug(
            f"{bgp_threshold=}, {nb_hostname_per_ns_org=}, {len(hostnames_per_config)} hostnames"
        )

        logger.info(f"Nb selected hostnames:: {len(selected_hostnames)}")

    logger.info(f"Total Nb selected hostnames:: {len(selected_hostnames)}")

    return selected_hostnames


async def resolve_subnets() -> None:

    if not END_TO_END_HOSTNAMES_PATH.exists():
        end_to_end_hostnames = load_hostnames()
        dump_csv(end_to_end_hostnames, END_TO_END_HOSTNAMES_PATH)

    if not END_TO_END_SUBNETS_PATH.exists():
        ripe_ip_map_subnets = load_ripe_ip_map_subnets()
        routers_subnets = load_routers_2ms_subnets()

        end_to_end_subnets = set(ripe_ip_map_subnets).union(set(routers_subnets))

        logger.info(f"RIPE IP map subnets:: {len(ripe_ip_map_subnets)} subnets")
        logger.info(f"Routers subnets:: {len(routers_subnets)} subnets")
        logger.info(f"End to end evaluation:: {len(end_to_end_subnets)} subnets")

        dump_json(ripe_ip_map_subnets, RIPE_IP_MAP_SUBNETS)
        dump_json(routers_subnets, ROUTERS_SUBNETS)
        dump_json(list(end_to_end_subnets), END_TO_END_SUBNETS_PATH)

    ripe_ip_map_targets = load_json(
        path_settings.END_TO_END_DATASET / "ripe_ip_map_targets_evaluation.json",
    )
    final_routers_evaluation = load_json(
        path_settings.END_TO_END_DATASET / "routers_targets_evaluation.json",
    )
    targets = set(ripe_ip_map_targets).union(final_routers_evaluation)

    target_subnets = set()
    for target in targets:
        subnet = get_prefix_from_ip(target)
        target_subnets.add(subnet)

    dump_json(list(target_subnets), END_TO_END_SUBNETS_PATH)

    cached_subnets = get_subnets("end_to_end_mapping_ecs")
    remaining_subnets = list(set(target_subnets).difference(set(cached_subnets)))

    logger.info(f"Missing subnets:: {len(remaining_subnets)}")

    dump_json(
        remaining_subnets, path_settings.END_TO_END_DATASET / "missing_subnets.json"
    )

    end_to_end_hostnames = load_csv(END_TO_END_HOSTNAMES_PATH)
    await resolve_vps_subnet(
        selected_hostnames=end_to_end_hostnames,
        input_file=path_settings.END_TO_END_DATASET / "missing_subnets.json",
        output_table=ECS_TABLE,
        chunk_size=100,
    )


def get_end_to_end_score() -> None:
    for bgp_threshold in [20, 50, 10]:
        for nb_hostname_per_ns_org in [3]:

            selected_hostnames_per_cdn_per_ns = load_json(
                path_settings.DATASET
                / f"hostname_geo_score_selection_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.json"
            )

            selected_hostnames = set()
            selected_hostnames_per_cdn = defaultdict(list)
            for ns in selected_hostnames_per_cdn_per_ns:
                for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
                    selected_hostnames.update(hostnames)
                    selected_hostnames_per_cdn[org].extend(hostnames)

            logger.info(
                f"{bgp_threshold=}, {nb_hostname_per_ns_org}, {len(selected_hostnames)=}"
            )

            output_path = (
                path_settings.RESULTS_PATH
                / f"end_to_end_evaluation/scores__best_hostname_geo_score_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.pickle"
            )

            # some organizations do not have enought hostnames
            if output_path.exists():
                logger.info(
                    f"Score for {bgp_threshold} BGP prefix threshold alredy exists"
                )
                continue

            score_config = {
                "targets_subnet_path": END_TO_END_SUBNETS_PATH,
                "vps_subnet_path": VPS_SUBNET_PATH,
                "hostname_per_cdn": selected_hostnames_per_cdn,
                "selected_hostnames": selected_hostnames,
                "targets_ecs_table": ECS_TABLE,
                "vps_ecs_table": VPS_ECS_MAPPING_TABLE,
                "hostname_selection": "max_bgp_prefix",
                "score_metric": ["jaccard"],
                "answer_granularities": ["answer_subnets"],
                "output_path": output_path,
            }

            get_scores(score_config)


def get_end_to_end_schedule(
    targets: list[str],
    subnet_scores: dict,
    vps_per_subnet: dict,
    last_mile_delay: dict,
    vps_coordinates: dict,
) -> dict:
    """get all remaining measurments for ripe ip map evaluation"""
    target_schedule = {}
    for target in tqdm(targets):
        target_subnet = get_prefix_from_ip(target)
        target_scores = subnet_scores[target_subnet]

        if not target_scores:
            logger.error(f"{target_subnet} does not have score")
        for _, target_score in target_scores.items():

            # get vps, function of their subnet ecs score
            ecs_vps = get_ecs_vps(
                target_subnet, target_score, vps_per_subnet, last_mile_delay, 5_00
            )

            # remove vps that have a high last mile delay
            ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

            ecs_vps = select_one_vp_per_as_city(
                ecs_vps, vps_coordinates, last_mile_delay
            )[:probing_budget]

            target_schedule[target] = [vp_addr for vp_addr, _ in ecs_vps]

    return target_schedule


def get_schedule_for_targets(
    targets: list[str], vps: list[str], vp_addr_per_id: dict
) -> dict:
    logger.info(
        f"Ping Schedule for:: {len(targets)} targets, {len(vps)} VPs per target"
    )

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    measurement_schedule = []
    for target in targets:
        for i in range(0, len(vps), 1):
            batch_vps = vps[i * batch_size : (i + 1) * batch_size]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target,
                    [vp_addr_per_id[vp["addr"]] for vp in batch_vps],
                )
            )

    return measurement_schedule


def get_measurement_schedule() -> dict[list]:
    """calculate distance error and latency for each score"""
    anycatch_db = load_anycatch_data()
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    ripe_ip_map_subnets = load_json(RIPE_IP_MAP_SUBNETS)
    ping_vps_to_target = get_pings_per_target(PING_RIPE_IP_MAP_TABLE)

    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    if not (
        path_settings.END_TO_END_DATASET / "ripe_ip_map_targets_evaluation.json"
    ).exists():

        ripe_ip_map_targets = []
        for target in ping_vps_to_target:
            target_subnet = get_prefix_from_ip(target)
            _, target_bgp_prefix = route_view_bgp_prefix(target, asndb)

            if target_subnet not in ripe_ip_map_subnets:
                continue

            if not target_bgp_prefix:
                continue

            if target_bgp_prefix in anycatch_db:
                continue

            if len(ripe_ip_map_targets) >= 1_100:
                break

            ripe_ip_map_targets.append(target)

        # routers dataset
        ecs_routers_subnets = load_json(ROUTERS_SUBNETS)
        routers_2ms_subnets = load_json(
            path_settings.END_TO_END_DATASET / "routers_2ms_subnets.json"
        )
        routers_2ms_targets = load_json(
            path_settings.END_TO_END_DATASET / "routers_2ms_targets.json"
        )
        routers_targets = load_csv(
            path_settings.END_TO_END_DATASET / "2500_random_routers_more_than_2ms.csv"
        )
        shuffle(routers_targets)

        routers_2ms_targets_evaluation = []
        for target in routers_2ms_targets:
            target_subnet = get_prefix_from_ip(target)
            if target_subnet in routers_2ms_subnets:
                routers_2ms_targets_evaluation.append(target)

            if len(routers_2ms_targets_evaluation) >= 800:
                break

        routers_targets_evaluation = []
        for target in routers_targets:
            target_subnet = get_prefix_from_ip(target)
            if target_subnet in ecs_routers_subnets:
                routers_targets_evaluation.append(target)

            if len(routers_targets_evaluation) >= 300:
                break

        final_routers_evaluation = list(
            set(routers_2ms_targets_evaluation).union(set(routers_targets_evaluation))
        )

        logger.info(
            f"Selected:: {len(ripe_ip_map_targets)} RIPE IP map IP addresses selected"
        )
        logger.info(
            f"Selected:: {len(final_routers_evaluation)} routers IP addresses selected"
        )

        # dump_json(
        #     ripe_ip_map_targets,
        #     path_settings.END_TO_END_DATASET / "ripe_ip_map_targets_evaluation.json",
        # )
        # dump_json(
        #     final_routers_evaluation,
        #     path_settings.END_TO_END_DATASET / "routers_targets_evaluation.json",
        # )

    ripe_ip_map_targets = load_json(
        path_settings.END_TO_END_DATASET / "ripe_ip_map_targets_evaluation.json",
    )
    final_routers_evaluation = load_json(
        path_settings.END_TO_END_DATASET / "routers_targets_evaluation.json",
    )

    targets = list(set(ripe_ip_map_targets).union(set(final_routers_evaluation)))

    logger.info(f"Ping schedule for:: {len(targets)}")
    measurement_schedule = get_schedule_for_targets(
        targets=targets,
        vps=vps,
        vp_addr_per_id=vp_id_per_addr,
    )

    return measurement_schedule


async def ping_targets(ping_from_cache: bool = True) -> None:
    """perfrom geolocation based on score similarity function"""
    if not ping_from_cache:
        measurement_schedule = get_measurement_schedule()
    else:
        # load previous measurements
        ping_vps_to_target = get_pings_per_target(PING_END_TO_END_TABLE)

        logger.info(f"Number of pings:: {len(ping_vps_to_target)}")
        vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
        vps_id_per_addr = {}
        for vp in vps:
            vps_id_per_addr[vp["addr"]] = vp["id"]

        for schedule_file in RIPEAtlasAPI().settings.MEASUREMENTS_SCHEDULE.iterdir():
            filtered_schedule = []
            if "ba1bc8e0-b28a-4184-87ad-1936777230c3" in schedule_file.name:
                logger.debug(f"Loading ping schedule:: {schedule_file.name}")

                # load schedule
                measurement_schedule = load_json(schedule_file)

                logger.info(
                    f"Size of original measurement schedule:: {len(measurement_schedule)}"
                )

                for target, vp_ids in tqdm(measurement_schedule):
                    measurement_done = False

                    try:
                        cached_pings = ping_vps_to_target[target]
                    except KeyError:
                        filtered_schedule.append((target, vp_ids))
                        continue

                    cached_vp_ids = []
                    for vp_addr, _ in cached_pings:
                        try:
                            cached_vp_ids.append(vps_id_per_addr[vp_addr])
                        except KeyError:
                            continue

                    for vp_id in vp_ids:
                        if vp_id in cached_vp_ids:
                            measurement_done = True
                            break

                    if not measurement_done:
                        filtered_schedule.append((target, vp_ids))

                measurement_schedule = filtered_schedule

                logger.info(
                    f"Size of filtered measurement schedule:: {len(measurement_schedule)}"
                )

    await RIPEAtlasProber(probing_type="pening", probing_tag="ping_end_to_end").main(
        measurement_schedule
    )


async def insert_measurements() -> None:
    cached_measurement_ids = await RIPEAtlasAPI().get_ping_measurement_ids(
        PING_END_TO_END_TABLE
    )

    measurement_ids = []
    for config_file in path_settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_end_to_end" in config_file.name:
            config = load_json(config_file)
            measurement_ids.extend([id for id in config["ids"]])

            logger.info(f"{config_file}:: {len(config['ids'])} ran")

    measurement_to_insert = list(
        set(measurement_ids).difference(set(cached_measurement_ids))
    )

    config_uuid = config["uuid"]
    logger.info(f"{len(measurement_to_insert)} measurements to insert")

    logger.info(
        f"Retreiving results for {len(measurement_to_insert)} measurements for {config_uuid=}"
    )

    batch_size = 100
    for i in range(0, len(measurement_to_insert), batch_size):
        ids = measurement_to_insert[i : i + batch_size]
        await retrieve_pings(ids, PING_END_TO_END_TABLE)
    # await retrieve_pings_from_tag(tag=config_uuid, output_table=PING_END_TO_END_TABLE)


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(PING_END_TO_END_TABLE, removed_vps)

    shortest_ping_per_target = []
    for target, pings in ping_vps_to_target.items():
        _, shortest_ping_rtt = min(pings, key=lambda x: x[-1])
        shortest_ping_per_target.append((target, shortest_ping_rtt))

    logger.info(f"{len(shortest_ping_per_target)} IP addresses remaining under 10ms")

    return shortest_ping_per_target


async def main() -> None:
    ecs_resoltion = False
    calculate_score = False
    geolocate = False
    insert = True
    evaluation = False

    if ecs_resoltion:
        await resolve_subnets()

    if calculate_score:
        get_end_to_end_score()

    if geolocate:
        await ping_targets()

    if insert:
        await insert_measurements()

    if evaluation:
        geo_resolver_sp = evaluate()
        plot_router_2ms(geo_resolver_sp, output_path="end_to_end_evaluation")


if __name__ == "__main__":
    asyncio.run(main())
