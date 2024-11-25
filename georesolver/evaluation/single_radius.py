"""this script aims at comparing the performances of georesolver against single-radius"""

import os
import httpx
import asyncio

from time import sleep
from pprint import pformat
from loguru import logger

from georesolver.agent import retrieve_pings
from georesolver.scheduler import create_agents
from georesolver.clickhouse.queries import get_pings_per_target, get_tables
from georesolver.common.files_utils import load_json, load_csv, dump_csv
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["RIPE_ATLAS_SECRET_KEY"] = (
    RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY_SECONDARY
)

NB_TARGETS = 100_000
SINGLE_RADIUS_TARGET_FILE = (
    path_settings.DATASET / "single_radius/single_radius_targets.csv"
)
SINGLE_RADIUS_MEASUREMENT_IDS = (
    path_settings.DATASET / "single_radius/single_radius_ids.csv"
)
SINGLE_RADIUS_PING_TABLE = "single_radius_ping"
GEORESOLVER_PING_TABLE = "single_radius_georesolver_ping"
GEORESOLVER_CONFIG_FILE = (
    path_settings.DATASET / "experiment_config/single_radius_config.json"
)


def get_single_radius_measurement_ids(wait_time: int = 30) -> None:
    """generate a file with single radius measurement ids"""
    tags = ["geolocation", "single-radius"]
    params = {"sort": ["-start_time"], "tags": tags, "status__in": "4", "af": 4}
    headers = {"Authorization": f"Key {RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY}"}

    logger.info("Retrieving measurement ids")
    logger.info(f"Number of measurement to retrive:: {NB_TARGETS}")

    measurement_ids = []
    with httpx.Client() as client:
        for _ in range(10):
            try:
                resp = client.get(
                    url=RIPEAtlasSettings().MEASUREMENT_URL,
                    params=params,
                    headers=headers,
                )
                resp = resp.json()
            except Exception:
                logger.info("Unsuported error, waiting")
                sleep(wait_time)
                continue

            if resp:
                break

        for measurement in resp["results"]:
            measurement_ids.append(measurement["id"])

        while len(measurement_ids) <= NB_TARGETS and resp["next"]:
            proggress = round((len(measurement_ids) / NB_TARGETS) * 100, 2)
            logger.info(
                f"Loading new page, {len(measurement_ids)} measurement retrieved {proggress}[%]"
            )
            for _ in range(10):
                try:
                    resp = client.get(url=resp["next"], headers=headers)
                    resp = resp.json()
                except Exception:
                    logger.info("Unsuported error, waiting")
                    sleep(wait_time)
                    continue
                if resp:
                    break
            for measurement in resp["results"]:
                measurement_ids.append(measurement["id"])

    dump_csv(measurement_ids, SINGLE_RADIUS_MEASUREMENT_IDS)


async def get_single_radius_measurement_results() -> None:
    """using previously retrieved measurement ids, get measurement results"""
    logger.info("Retrieving measurement results")

    measurement_ids = load_csv(SINGLE_RADIUS_MEASUREMENT_IDS)

    batch_size = 1_000
    for i in range(0, len(measurement_ids), batch_size):
        batch_ids = measurement_ids[i : i + batch_size]
        logger.info(f"batch:: {i} -> {i + batch_size}")
        await retrieve_pings(
            ids=batch_ids,
            output_table=SINGLE_RADIUS_PING_TABLE,
        )


def generate_target_file() -> None:
    """generate target file based on retrieved measurement results"""
    pings_per_target = get_pings_per_target(table_name=SINGLE_RADIUS_PING_TABLE)
    targets = [t for t in pings_per_target]
    dump_csv(targets, SINGLE_RADIUS_TARGET_FILE)


def load_dataset() -> None:
    """retrieve measurements using single radius tag"""
    if not SINGLE_RADIUS_MEASUREMENT_IDS.exists():
        get_single_radius_measurement_ids()

    if not SINGLE_RADIUS_PING_TABLE in get_tables():
        asyncio.run(get_single_radius_measurement_results())

    if not SINGLE_RADIUS_TARGET_FILE.exists():
        generate_target_file()


def run_georesolver() -> None:
    """perform georesolver measurement on single radius dataset"""
    config = load_json(GEORESOLVER_CONFIG_FILE)
    logger.info("Running GeoResolver with config:: ")
    logger.info("config=\n{}", pformat(config))

    agents = create_agents(GEORESOLVER_CONFIG_FILE)
    for agent in agents:
        agent.run()


def run_evaluation() -> None:
    """
    perform the evaluation:
        - latency comparison overall
        - AS diversity
        - VPs selection analysis (to define)
    """
    pass


def main() -> None:
    """
    main function:
        - generate a dataset of single radius measurements
        - perform georesolver measurement on the dataset
        - make the evaluation:
            - latency comparison overall
            - AS diversity
            - VPs selection analysis (to define)
    """
    perform_measurement: bool = True
    make_eval: bool = False
    if perform_measurement:
        load_dataset()
        run_georesolver()

    if make_eval:
        run_evaluation()


if __name__ == "__main__":
    main()
