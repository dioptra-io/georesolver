import asyncio

from random import shuffle
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pych_client import ClickHouseClient
from ipaddress import IPv4Address, AddressValueError


from geogiant.clickhouse import GetSubnets, GetDstPrefix
from geogiant.prober import RIPEAtlasAPI, RIPEAtlasProber
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.evaluation.plot import plot_router_2ms
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    load_vps,
    retrieve_pings,
    get_pings_per_target,
)
from geogiant.common.utils import TargetScores, get_parsed_vps
from geogiant.common.files_utils import (
    load_csv,
    dump_csv,
    load_json,
    dump_json,
    load_pickle,
    load_json_iter,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# INPUT TABLE
PING_RIPE_IP_MAP_TABLE = "pings_ripe_ip_map"

# OUTPUT TABLES
PING_END_TO_END_TABLE = "pings_end_to_end"
ECS_TABLE = "end_to_end_mapping_ecs"
VPS_ECS_TABLE = "vps_mapping_ecs"

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
END_TO_END_SUBNETS_PATH = (
    path_settings.END_TO_END_DATASET / "end_to_end_subnet_evaluation.json"
)
VPS_SUBNET_PATH = path_settings.END_TO_END_DATASET / "vps_subnet.json"
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

    end_to_end_hostnames = load_csv(END_TO_END_HOSTNAMES_PATH)
    await resolve_vps_subnet(
        selected_hostnames=end_to_end_hostnames,
        input_file=END_TO_END_SUBNETS_PATH,
        output_table=ECS_TABLE,
        chunk_size=100,
    )


def get_end_to_end_score() -> None:
    # END_TO_END_SUBNETS_PATH.unlink()

    for bgp_threshold in [10, 20, 50]:

        selected_hostnames_per_cdn = load_json(
            path_settings.DATASET
            / f"hostname_geo_score_selection_{bgp_threshold}_BGP.json"
        )

        SCORE_FILE = (
            path_settings.RESULTS_PATH
            / f"end_to_end_evaluation/scores__best_hostname_geo_score_{bgp_threshold}.pickle"
        )

        selected_hostnames = set()
        for org, hostnames in selected_hostnames_per_cdn.items():
            logger.info(f"{org=}, {len(hostnames)=}")
            selected_hostnames.update(hostnames)

        logger.info(f"{len(selected_hostnames)=}")

        for org, hostnames in selected_hostnames_per_cdn.items():
            logger.info(f"{org=}, {len(hostnames)=}")

        score_config = {
            "targets_subnet_path": END_TO_END_SUBNETS_PATH,
            "VPS_SUBNET_PATH": VPS_SUBNET_PATH,
            "hostname_per_cdn": selected_hostnames_per_cdn,
            "selected_hostnames": selected_hostnames,
            "targets_ECS_TABLE": ECS_TABLE,
            "VPS_ECS_TABLE": VPS_ECS_TABLE,
            "hostname_selection": "max_bgp_prefix",
            "score_metric": [
                "jaccard",
            ],
            "answer_granularities": [
                "answer_subnets",
            ],
            "output_path": SCORE_FILE,
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


def get_measurement_schedule(max_nb_measurements: int = 10_000) -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)

    target_subnets = get_end_to_end_subnets()
    targets = load_targets(target_subnets)

    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    vps_per_subnet, vps_coordinates = get_parsed_vps(
        vps, asndb, removed_vps=removed_vps
    )

    logger.info("BGP prefix score geoloc evaluation")

    scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / SCORE_FILE)
    scores = scores.score_answer_subnets

    logger.info(f"Score retrieved for:: {len(scores)}")
    logger.info(f"Ping schedule for:: {len(targets)}")

    measurement_schedule = get_end_to_end_schedule(
        targets=targets,
        subnet_scores=scores,
        vps_per_subnet=vps_per_subnet,
        last_mile_delay=last_mile_delay,
        vps_coordinates=vps_coordinates,
    )

    ping_schedule_with_id = []
    for target, selected_vps in measurement_schedule.items():

        # find vp id
        ping_schedule_with_id.append(
            (target, [vp_id_per_addr[vp_addr] for vp_addr in selected_vps])
        )

    # randomly select some IP addresses to geolocate
    if len(ping_schedule_with_id) > max_nb_measurements:
        shuffle(ping_schedule_with_id)
        ping_schedule_with_id = ping_schedule_with_id[:max_nb_measurements]

    logger.info(f"Ping schedule for:: {len(ping_schedule_with_id)}")

    return ping_schedule_with_id


async def ping_targets() -> None:
    """perfrom geolocation based on score similarity function"""
    measurement_schedule = get_measurement_schedule()

    await RIPEAtlasProber(probing_type="ping", probing_tag="ping_end_to_end").main(
        measurement_schedule
    )


async def insert_measurements() -> None:
    # retrive ping measurements and insert them into clickhouse db
    measurement_ids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_end_to_end" in config_file.name:
            config = load_json(config_file)
            measurement_ids.extend([id for id in config["ids"]])

    logger.info(f"Retreiving results for {len(measurement_ids)} measurements")

    await retrieve_pings(measurement_ids, PING_END_TO_END_TABLE)


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
    # retrieve all IP addresses with < 2ms
    # analysis
    # score
    # Ping schedule + measurement
    ecs_resoltion = True
    calculate_score = False
    geolocate = False
    insert = False
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
