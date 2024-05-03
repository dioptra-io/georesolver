import asyncio

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
from geogiant.common.queries import get_min_rtt_per_vp, load_vps, retrieve_pings
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
ping_table = "pings_routers_2ms"
ecs_table = "routers_2ms_mapping_ecs"
vps_ecs_table = "vps_mapping_ecs"

# FILE PATHS
filtered_routers_2ms_targets_path = (
    path_settings.DATASET / "filtered_routers_2ms_targets.json"
)
routers_2ms_targets_path = path_settings.DATASET / "routers_2ms_targets.json"
routers_2ms_subnets_path = path_settings.DATASET / "routers_2ms_subnet_evaluation.json"
vps_subnet_path = path_settings.DATASET / "vps_subnet.json"
score_file = (
    path_settings.RESULTS_PATH
    / f"routers_2ms_evaluation/scores__best_hostname_geo_score.pickle"
)
results_file = (
    path_settings.RESULTS_PATH
    / f"routers_2ms_evaluation/{'results' + str(score_file).split('scores')[-1]}"
)

# CONSTANT PARAMETERS
probing_budget = 50


def generate_routers_targets_file() -> None:
    if not (routers_2ms_targets_path).exists():
        rows = load_json_iter(path_settings.DATASET / "routers_2ms.json")
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

        dump_json(targets, routers_2ms_targets_path)


def load_targets(target_subnets: list) -> None:

    if not filtered_routers_2ms_targets_path.exists():
        filtered_targets = []

        generate_routers_targets_file()
        all_targets = load_json(routers_2ms_targets_path)

        for target_addr, target_info in all_targets.items():
            if not target_info["subnet"] in target_subnets:
                continue

            filtered_targets.append(target_addr)

        dump_json(filtered_targets, filtered_routers_2ms_targets_path)

    filtered_targets = load_json(filtered_routers_2ms_targets_path)

    return filtered_targets


def get_routers_2ms_subnets() -> list:
    """retrieve all RIPE IP map subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetSubnets().execute(client=client, table_name=ecs_table)

    routers_2ms_subnets = []
    for row in rows:
        routers_2ms_subnets.append(row["subnet"])

    return routers_2ms_subnets


def get_subnets() -> list[str]:
    if routers_2ms_subnets_path.exists():
        routers_2ms_subnets = get_routers_2ms_subnets()

        if not routers_2ms_subnets:
            raise RuntimeError(
                f"No routers subents found in clickhouse table:: {ecs_table}"
            )

        logger.info(f"Retrieved:: {len(routers_2ms_subnets)} subnets")

        dump_json(routers_2ms_subnets, routers_2ms_subnets_path)


async def resolve_subnets() -> None:
    selected_hostnames = load_csv(
        path_settings.DATASET / "selected_hostname_geo_score.csv"
    )

    selected_hostnames = load_csv(path_settings.DATASET / "missing_hostnames.csv")

    logger.info(f"Performing resolution on:: {len(selected_hostnames)}")

    get_subnets()

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=routers_2ms_subnets_path,
        output_table=ecs_table,
        chunk_size=100,
    )


def get_routers_2ms_score() -> None:
    # routers_2ms_subnets_path.unlink()

    for bgp_threshold in [10, 20, 50]:

        selected_hostnames_per_cdn = load_json(
            path_settings.DATASET
            / f"hostname_geo_score_selection_{bgp_threshold}_BGP.json"
        )

        score_file = (
            path_settings.RESULTS_PATH
            / f"routers_2ms_evaluation/scores__best_hostname_geo_score_{bgp_threshold}.pickle"
        )

        selected_hostnames = set()
        for org, hostnames in selected_hostnames_per_cdn.items():
            logger.info(f"{org=}, {len(hostnames)=}")
            selected_hostnames.update(hostnames)

        logger.info(f"{len(selected_hostnames)=}")

        for org, hostnames in selected_hostnames_per_cdn.items():
            logger.info(f"{org=}, {len(hostnames)=}")

        score_config = {
            "targets_subnet_path": routers_2ms_subnets_path,
            "vps_subnet_path": vps_subnet_path,
            "hostname_per_cdn": selected_hostnames_per_cdn,
            "selected_hostnames": selected_hostnames,
            "targets_ecs_table": ecs_table,
            "vps_ecs_table": vps_ecs_table,
            "hostname_selection": "max_bgp_prefix",
            "score_metric": [
                "jaccard",
            ],
            "answer_granularities": [
                "answer_subnets",
            ],
            "output_path": score_file,
        }

        get_scores(score_config)


def get_routers_2ms_schedule(
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

    target_subnets = get_routers_2ms_subnets()
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

    measurement_schedule = get_routers_2ms_schedule(
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

    await RIPEAtlasProber(probing_type="ping", probing_tag="ping_routers_2ms").main(
        measurement_schedule
    )


async def insert_measurements() -> None:
    # retrive ping measurements and insert them into clickhouse db
    measurement_ids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_routers_2ms" in config_file.name:
            config = load_json(config_file)
            measurement_ids.extend([id for id in config["ids"]])

    logger.info(f"Retreiving results for {len(measurement_ids)} measurements")

    await retrieve_pings(measurement_ids, ping_table)


async def main() -> None:
    # retrieve all IP addresses with < 2ms
    # analysis
    # score
    # Ping schedule + measurement
    ecs_resoltion = False
    calculate_score = True
    geolocate = False
    insert = False

    if ecs_resoltion:
        await resolve_subnets()

    if calculate_score:
        get_routers_2ms_score()

    if geolocate:
        await ping_targets()

    if insert:
        await insert_measurements()


if __name__ == "__main__":
    asyncio.run(main())
