import asyncio

from collections import defaultdict
from random import shuffle
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from pych_client import ClickHouseClient
from ipaddress import IPv4Address, AddressValueError


from geogiant.clickhouse import GetSubnets
from geogiant.prober import RIPEAtlasAPI, RIPEAtlasProber
from geogiant.hostname_init import resolve_vps_subnet
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
    load_json,
    dump_json,
    load_pickle,
    load_json_iter,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# CLICKHOUSE TABLE
PING_ROUTER_TABLE = "pings_routers"
ECS_TABLE = "end_to_end_mapping_ecs"
VPS_ECS_TABLE = "vps_mapping_ecs"

# FILE PATHS
ROUTERS_TARGET_PATH = path_settings.DATASET / "routers_targets.json"
ROUTERS_SUBNET_PATH = path_settings.END_TO_END_DATASET / "routers_subnets.json"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"

# CONSTANT PARAMETERS
probing_budget = 50


def generate_routers_targets_file() -> None:
    if not (ROUTERS_TARGET_PATH).exists():
        rows = load_json_iter(path_settings.DATASET / "routers.json")
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

        dump_json(targets, ROUTERS_TARGET_PATH)


def get_routers_subnets() -> list:
    """retrieve all RIPE IP map subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetSubnets().execute(client=client, table_name=ECS_TABLE)

    routers_subnets = []
    for row in rows:
        routers_subnets.append(row["subnet"])

    return routers_subnets


def get_subnets() -> list[str]:
    if ROUTERS_SUBNET_PATH.exists():
        routers_subnets = get_routers_subnets()

        if not routers_subnets:
            raise RuntimeError(
                f"No routers subents found in clickhouse table:: {ECS_TABLE}"
            )

        logger.info(f"Retrieved:: {len(routers_subnets)} subnets")

        dump_json(routers_subnets, ROUTERS_SUBNET_PATH)


async def resolve_subnets() -> None:
    selected_hostnames = load_csv(
        path_settings.DATASET / "selected_hostname_geo_score.csv"
    )

    selected_hostnames = load_csv(path_settings.DATASET / "missing_hostnames.csv")

    logger.info(f"Performing resolution on:: {len(selected_hostnames)}")

    get_subnets()

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=ROUTERS_SUBNET_PATH,
        output_table=ECS_TABLE,
        chunk_size=100,
    )


def get_routers_score() -> None:
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
                / f"routers_evaluation/scores__best_hostname_geo_score_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.pickle"
            )

            # some organizations do not have enought hostnames
            if output_path.exists():
                logger.info(
                    f"Score for {bgp_threshold} BGP prefix threshold alredy done"
                )
                continue

            score_config = {
                "targets_subnet_path": ROUTERS_SUBNET_PATH,
                "vps_subnet_path": VPS_SUBNET_PATH,
                "hostname_per_cdn": selected_hostnames_per_cdn,
                "selected_hostnames": selected_hostnames,
                "targets_ecs_table": ECS_TABLE,
                "vps_ecs_table": VPS_ECS_TABLE,
                "hostname_selection": "max_bgp_prefix",
                "score_metric": ["jaccard"],
                "answer_granularities": ["answer_subnets"],
                "output_path": output_path,
            }

            get_scores(score_config)


def get_routers_schedule(
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

    target_subnets = get_routers_subnets()
    targets = load_targets(target_subnets)

    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    vps_per_subnet, vps_coordinates = get_parsed_vps(
        vps, asndb, removed_vps=removed_vps
    )

    logger.info("BGP prefix score geoloc evaluation")

    scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)
    scores = scores.score_answer_subnets

    logger.info(f"Score retrieved for:: {len(scores)}")
    logger.info(f"Ping schedule for:: {len(targets)}")

    measurement_schedule = get_routers_schedule(
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

    await RIPEAtlasProber(probing_type="ping", probing_tag="ping_routers").main(
        measurement_schedule
    )


async def insert_measurements() -> None:
    # retrive ping measurements and insert them into clickhouse db
    measurement_ids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_routers" in config_file.name:
            config = load_json(config_file)
            measurement_ids.extend([id for id in config["ids"]])

    logger.info(f"Retreiving results for {len(measurement_ids)} measurements")

    await retrieve_pings(measurement_ids, PING_ROUTER_TABLE)


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(PING_ROUTER_TABLE, removed_vps)

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
    ecs_resoltion = False
    calculate_score = True
    geolocate = False
    insert = False
    evaluation = False

    if ecs_resoltion:
        await resolve_subnets()

    if calculate_score:
        get_routers_score()

    if geolocate:
        await ping_targets()

    if insert:
        await insert_measurements()

    if evaluation:
        geo_resolver_sp = evaluate()
        plot_routers(geo_resolver_sp, output_path="routers_evaluation")


if __name__ == "__main__":
    asyncio.run(main())
