"""run GeoResolver geolocation on a set of demo targets"""

from random import sample
from loguru import logger

from georesolver.georesolver import run_georesolver
from georesolver.evaluation.itdk_dataset import load_csv, dump_csv
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

DEMO_TARGET_FILE = path_settings.DATASET / "demo_targets.csv"
CONFIG_PATH = path_settings.DATASET / "experiment_config/local_demo_config.json"
RANDOM_DATASET = True
NB_ADDRS = 1_000


def generate_random_dataset() -> None:
    """generate a random target IP addresses, output into target file"""
    ip_addrs = load_csv(
        path_settings.DATASET / "subnet_aggregation/subnet_aggregation_targets.csv"
    )
    sample_targets = sample(ip_addrs, NB_ADDRS)
    dump_csv(sample_targets, DEMO_TARGET_FILE)


def main_demo() -> None:
    """run georesolver on a random subset of ITDK IP addresses"""

    if RANDOM_DATASET:
        # dump N random address from ITDK dataset in demo file
        logger.info(f"Generating {NB_ADDRS} random target file for demo")
        generate_random_dataset()

    logger.info(f"File available at:: {DEMO_TARGET_FILE}")

    run_georesolver(CONFIG_PATH.resolve())


if __name__ == "__main__":
    main_demo()
