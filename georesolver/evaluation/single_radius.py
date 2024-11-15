"""this script aims at comparing the performances of georesolver against single-radius"""

import httpx

from time import sleep
from pprint import pprint
from loguru import logger

from georesolver.common.files_utils import dump_csv
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)


path_settings = PathSettings()
ch_settings = ClickhouseSettings()

SINGLE_RADIUS_TARGET_FILE = (
    path_settings.DATASET / "single_radius/single_radius_targets.csv"
)
SINGLE_RADIUS_MEASUREMENT_IDS = (
    path_settings.DATASET / "single_radius/single_radius_ids.csv"
)
SINGLE_RADIUS_PING_TABLE = "single_radius_ping"
GEORESOLVER_PING_TABLE = "single_radius_georesolver_ping"


def get_single_radius_measurement_ids(
    nb_measurements: int = 10,
    wait_time: int = 30,
) -> None:
    """generate a file with single radius measurement ids"""
    tags = ["geolocation", "single-radius"]
    params = {"sort": ["-start_time"], "tags": tags, "status": "Stopped"}
    headers = {"Authorization": f"Key {RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY}"}

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

        while len(measurement_ids) <= nb_measurements and measurement["next"]:
            for _ in range(10):
                try:
                    resp = client.get(url=measurement["next"], headers=headers)
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


def get_single_radius_measurement_results() -> None:
    """using previously retrieved measurement ids, get measurement results"""


def load_dataset() -> None:
    """retrieve measurements using single radius tag"""
    if not SINGLE_RADIUS_MEASUREMENT_IDS.exists():
        get_single_radius_measurement_ids()

    if not SINGLE_RADIUS_TARGET_FILE.exists():
        get_single_radius_measurement_results()


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
    load_dataset()


if __name__ == "__main__":
    main()
