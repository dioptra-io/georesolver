import asyncio
import time

from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm

from geogiant.prober import RIPEAtlasAPI, RIPEAtlasProber
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.evaluation.plot import plot_internet_scale
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.utils import TargetScores, get_parsed_vps
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    load_vps,
    retrieve_pings,
    get_pings_per_target,
    get_dst_prefix,
)
from geogiant.common.files_utils import (
    load_pickle,
    load_json,
    dump_pickle,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


PING_TABLE = "pings_internet_scale"
ECS_TABLE = "internet_scale_mapping_ecs"
VPS_ECS_TABLE = "vps_mapping_ecs"

PROBING_BUDGET = 50

INTERNET_SCALE_SUBNETS_PATH = (
    path_settings.INTERNET_SCALE_DATASET / "all_internet_scale_subnets.json"
)
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"
INTERNET_SCALE_EVALUATION_PATH = (
    path_settings.RESULTS_PATH
    / "internet_scale_evaluation/results__routers_no_users.pickle"
)


def get_geo_resolver_schedule(
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
            )[:PROBING_BUDGET]

            target_schedule[target] = [vp_addr for vp_addr, _ in ecs_vps]

    return target_schedule


def load_subnet_scores() -> list[str]:
    cached_subnets = get_dst_prefix(PING_TABLE)
    subnet_scores = {}
    for score_file in path_settings.INTERNET_SCALE_RESULTS.iterdir():
        if "score__" in score_file.name:
            score: TargetScores = load_pickle(score_file)
            score = score.score_answer_subnets
            score_subnets = [subnet for subnet in score]

            logger.debug(f"Score subnets for:: {len(score_subnets)} subnets")

            subnet_schedule = set(score_subnets).difference(set(cached_subnets))

            if len(subnet_schedule) < len(score_subnets):
                logger.debug(f"Subnets already geolocated")
                continue

            for subnet in subnet_schedule:
                subnet_scores[subnet] = score[subnet]

        if len(subnet_scores) > 10_000:
            logger.info("Enought subnets loaded")
            break

    return subnet_scores


def load_targets_from_score(
    subnet_scores: dict[list], addr_per_subnet: dict
) -> list[str]:
    targets = []
    for target_subnet in subnet_scores:
        target_addr = addr_per_subnet[target_subnet][0]
        targets.append(target_addr)

    return targets


def get_measurement_schedule() -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )

    internet_scale_dataset = load_json(path_settings.USER_HITLIST_FILE)
    subnet_scores = load_subnet_scores()
    targets = load_targets_from_score(subnet_scores, internet_scale_dataset)

    logger.info(f"Targets in schedule:: {len(targets)}")

    target_schedule = get_geo_resolver_schedule(
        targets,
        subnet_scores,
        vps_per_subnet,
        last_mile_delay,
        vps_coordinates,
    )

    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    measurement_schedule = []
    for target, vps in target_schedule.items():
        measurement_schedule.append(
            (target, [vp_id_per_addr[vp_addr] for vp_addr in vps])
        )

    return measurement_schedule


async def ping_targets(wait_time: int = 60 * 20) -> None:
    """perfrom geolocation based on score similarity function"""
    while True:
        await insert_measurements()
        measurement_schedule = get_measurement_schedule()

        if measurement_schedule:

            batch_size = 1_000
            for i in range(0, len(measurement_schedule), batch_size):

                batch_schedule = measurement_schedule[i : i + batch_size]

                logger.debug(
                    f"Pings schedule (batch {(i + 1) // batch_size}/{len(measurement_schedule) // batch_size})"
                )
                await RIPEAtlasProber(
                    probing_type="ping", probing_tag="ping_internet_scale"
                ).main(batch_schedule)

                await asyncio.sleep(60 * 2)

                await insert_measurements()

        else:
            logger.info(f"No measurement available, pause: {wait_time} mins")
            await asyncio.sleep(wait_time)


async def insert_measurements() -> None:
    cached_measurement_ids = await RIPEAtlasAPI().get_ping_measurement_ids(PING_TABLE)

    measurement_ids = []
    config_uuids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_internet_scale" in config_file.name:
            config = load_json(config_file)

            if config["ids"]:
                measurement_ids.extend(config["ids"])
                logger.info(f"{config_file}:: {len(config['ids'])} ran")
                config_uuids.append(config["uuid"])

    measurement_to_insert = list(
        set(measurement_ids).difference(set(cached_measurement_ids))
    )

    logger.info(f"{len(measurement_to_insert)} measurements to insert")

    logger.info(f"Retreiving results for {len(measurement_to_insert)} measurements")

    batch_size = 100
    for i in range(0, len(measurement_to_insert), batch_size):
        ids = measurement_to_insert[i : i + batch_size]
        await retrieve_pings(ids, PING_TABLE)


def get_shortest_ping_all_vp(ping_table: str, removed_vps: list[str] = []) -> None:
    """for each IP addresses retrieved the shortest ping"""
    ping_vps_to_target = get_pings_per_target(ping_table, removed_vps)

    shortest_ping_per_target = []
    for target_addr, target_pings in tqdm(ping_vps_to_target.items()):

        _, shortest_ping_rtt = min(target_pings, key=lambda x: x[-1])
        shortest_ping_per_target.append((target_addr, shortest_ping_rtt))

    return shortest_ping_per_target


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    removed_vps = load_json(path_settings.REMOVED_VPS)

    ping_vps_to_target = get_pings_per_target(PING_TABLE, removed_vps)

    shortest_ping_per_target = []
    for target, pings in ping_vps_to_target.items():
        _, shortest_ping_rtt = min(pings, key=lambda x: x[-1])
        shortest_ping_per_target.append((target, shortest_ping_rtt))

    logger.info(f"{len(shortest_ping_per_target)} IP addresses")

    dump_pickle(shortest_ping_per_target, INTERNET_SCALE_EVALUATION_PATH)

    return shortest_ping_per_target


async def main() -> None:
    geolocate = True
    evaluation = True
    make_figures = True

    if geolocate:
        await ping_targets()

    if evaluation:
        evaluate()

    if make_figures:
        geo_resolver_sp = load_pickle(INTERNET_SCALE_EVALUATION_PATH)
        plot_internet_scale(
            geo_resolver_sp=geo_resolver_sp, output_path="internet_scale_evaluation"
        )


if __name__ == "__main__":
    asyncio.run(main())
