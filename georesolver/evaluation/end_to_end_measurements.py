"""perform ECS resolution, score calculation and pings mesehd pings for routers and RIPE IP map datasets"""

import asyncio

from tqdm import tqdm
from random import shuffle
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from pych_client import ClickHouseClient
from ipaddress import IPv4Address, AddressValueError


from georesolver.clickhouse import GetDstPrefix
from georesolver.prober import RIPEAtlasAPI, RIPEAtlasProber
from georesolver.agent import retrieve_pings
from georesolver.agent import run_dns_mapping
from georesolver.evaluation.evaluation_score_functions import main_score
from georesolver.evaluation.evaluation_ecs_geoloc_functions import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.clickhouse.queries import (
    load_vps,
    get_pings_per_target,
    get_subnets,
)
from georesolver.common.files_utils import (
    load_csv,
    load_json,
    dump_json,
    load_json_iter,
    load_anycatch_data,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# INPUT TABLE
PING_RIPE_IP_MAP_TABLE = "pings_single_radius"

# OUTPUT TABLES
PING_END_TO_END_TABLE = "pings_end_to_end"
ECS_TABLE = "end_to_end_ecs_mapping"
VPS_ECS_MAPPING_TABLE = "vps_ecs_mapping"

# FILE PATHS
RIPE_IP_MAP_SUBNETS = path_settings.END_TO_END_DATASET / "ripe_ip_map_subnets.json"
ROUTERS_SUBNETS = path_settings.END_TO_END_DATASET / "routers_subnets.json"
END_TO_END_HOSTNAMES_PATH = path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv"
filtered_end_to_end_targets_path = (
    path_settings.END_TO_END_DATASET / "filtered_end_to_end_targets.json"
)
end_to_end_targets_path = path_settings.END_TO_END_DATASET / "end_to_end_targets.json"
END_TO_END_SUBNETS_PATH = path_settings.END_TO_END_DATASET / "end_to_end_subnets.json"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"
RESULT_FILE = path_settings.RESULTS_PATH / "end_to_end_evaluation/results.pickle"

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

    cached_subnets = get_subnets("end_to_end_ecs_mapping")
    remaining_subnets = list(set(target_subnets).difference(set(cached_subnets)))

    logger.info(f"Missing subnets:: {len(remaining_subnets)}")

    dump_json(
        remaining_subnets, path_settings.END_TO_END_DATASET / "missing_subnets.json"
    )

    end_to_end_hostnames = load_csv(END_TO_END_HOSTNAMES_PATH)
    await run_dns_mapping(
        selected_hostnames=end_to_end_hostnames,
        input_file=path_settings.END_TO_END_DATASET / "missing_subnets.json",
        output_table=ECS_TABLE,
    )


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


async def ping_targets(ping_from_cache: bool = True, tag: str = None) -> None:
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
            if tag in schedule_file.name:
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

    if ecs_resoltion:
        await resolve_subnets()

    if calculate_score:
        main_score(
            target_subnet_path=END_TO_END_SUBNETS_PATH,
            vps_subnet_path=VPS_SUBNET_PATH,
            ecs_table=ECS_TABLE,
            vps_ecs_table=VPS_ECS_MAPPING_TABLE,
            hostname_file_path=path_settings.HOSTNAME_FILES
            / "hostnames_georesolver.csv",
            output_path=path_settings.RESULTS_PATH / "scores.pickle",
        )

    if geolocate:
        latest_tag = "ba1bc8e0-b28a-4184-87ad-1936777230c3"
        await ping_targets(ping_from_cache=True, latest_tag=latest_tag)

    if insert:
        await insert_measurements()


if __name__ == "__main__":
    asyncio.run(main())
