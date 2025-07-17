"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger

from georesolver.georesolver import run_georesolver
from georesolver.evaluation.itdk_dataset_parsing import get_random_itdk_routers
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/remote_example_config.json"
DEMO_TARGET_FILE = path_settings.DATASET / "demo_targets.csv"
REGENERATE_FILE = True
NB_ADDRS = 10


def main_demo() -> None:
    """run georesolver on a random subset of ITDK IP addresses"""
    if REGENERATE_FILE:
        # dump N random address in demo file
        logger.info(f"Generating {NB_ADDRS} random target file for demo")
        get_random_itdk_routers(
            NB_ADDRS,
            DEMO_TARGET_FILE,
            mode="w",
        )
        logger.info(f"File available at:: {DEMO_TARGET_FILE}")

    # debugging
    run_georesolver(CONFIG_PATH)


if __name__ == "__main__":
    main_demo()
