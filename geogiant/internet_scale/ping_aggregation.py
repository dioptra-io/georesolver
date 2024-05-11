import asyncio
import subprocess

from random import shuffle
from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from geogiant.prober import RIPEAtlasAPI, RIPEAtlasProber
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.internet_scale.zmap import zmap, get_shortest_ping_vps
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
    load_csv,
    dump_csv,
    load_json,
    dump_json,
    create_tmp_csv_file,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


PING_GEORESOLVER_TABLE = "pings_internet_scale"
PING_AGGREGATION_TABLE = "pings_internet_scale_aggregation"

PROBING_BUDGET = 50


INTERNET_SCALE_SUBNETS_PATH = (
    path_settings.INTERNET_SCALE_DATASET / "all_internet_scale_subnets.json"
)
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnets_filtered.json"
INTERNET_SCALE_EVALUATION_PATH = (
    path_settings.RESULTS_PATH
    / "internet_scale_evaluation/results__routers_internet_scale.pickle"
)
INTERNET_SCALE_RESPONSIVE_IP_ADDRS = (
    path_settings.INTERNET_SCALE_DATASET / "responsive_ip_addrs_per_subnet.json"
)


def load_targets_from_score(
    subnet_scores: dict[list], addr_per_subnet: dict
) -> list[str]:
    targets = []
    for target_subnet in subnet_scores:
        target_addr = addr_per_subnet[target_subnet][0]
        targets.append(target_addr)

    return targets


async def ping_targets(wait_time: int = 60 * 20) -> dict[list]:
    """calculate distance error and latency for each score"""
    # check if some measurements are ready to be inserted

    while True:
        await insert_measurements()

        cached_pings = get_shortest_ping_vps(PING_AGGREGATION_TABLE)
        measurement_schedule = load_json(INTERNET_SCALE_RESPONSIVE_IP_ADDRS)

        filtered_schedule = []
        for target, vp_ids in measurement_schedule:
            if target in cached_pings:
                continue
            filtered_schedule.append((target, vp_ids))

        if measurement_schedule:

            batch_size_pings = 1_000
            for i in range(0, len(measurement_schedule), batch_size_pings):

                batch_schedule = measurement_schedule[i : i + batch_size_pings]

                logger.debug(
                    f"Pings schedule (batch {(i + 1) // batch_size_pings}/{len(measurement_schedule) // batch_size_pings})"
                )
                await RIPEAtlasProber(
                    probing_type="ping", probing_tag="ping_per_subnets"
                ).main(batch_schedule)

                await insert_measurements()

            else:
                logger.info(f"No measurement available, pause: {wait_time} mins")
                asyncio.sleep(wait_time)


async def insert_measurements() -> None:
    cached_measurement_ids = await RIPEAtlasAPI().get_ping_measurement_ids(
        PING_AGGREGATION_TABLE
    )

    measurement_ids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_per_subnets" in config_file.name:
            config = load_json(config_file)
            if config["ids"]:
                measurement_ids.extend([id for id in config["ids"]])

                logger.info(f"{config_file}:: {len(config['ids'])} ran")

    measurement_to_insert = list(
        set(measurement_ids).difference(set(cached_measurement_ids))
    )

    logger.info(f"{len(measurement_to_insert)} measurements to insert")

    logger.info(f"Retreiving results for {len(measurement_to_insert)} measurements")

    batch_size = 100
    for i in range(0, len(measurement_to_insert), batch_size):
        ids = measurement_to_insert[i : i + batch_size]
        await retrieve_pings(ids, PING_AGGREGATION_TABLE)


async def main() -> None:
    await ping_targets()


if __name__ == "__main__":
    asyncio.run(main())
