"""run GeoResolver geolocation on a set of demo targets"""

from random import sample
from loguru import logger

from georesolver.scheduler import create_agents
from georesolver.evaluation.itdk_dataset import (
    get_random_itdk_routers,
    load_csv,
    dump_csv,
)
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/local_demo_config.json"
DEMO_TARGET_FILE = path_settings.DATASET / "demo_targets.csv"
DEMO_ECS_MAPPING_TABLE = "demo_ecs_mapping"
NB_ADDRS = 10_000


def generate_random_dataset() -> None:
    """generate a random target IP addresses, output into target file"""
    ip_addrs = load_csv(
        path_settings.DATASET / "subnet_aggregation/subnet_aggregation_targets.csv"
    )

    sample_targets = sample(ip_addrs, NB_ADDRS)

    dump_csv(sample_targets, DEMO_TARGET_FILE)


def main_demo() -> None:
    """run georesolver on a random subset of ITDK IP addresses"""
    # dump N random address from ITDK dataset in demo file
    logger.info(f"Generating {NB_ADDRS} random target file for demo")
    generate_random_dataset()
    # get_random_itdk_routers(
    #     NB_ADDRS,
    #     DEMO_TARGET_FILE,
    #     mode="w",
    # )
    logger.info(f"File available at:: {DEMO_TARGET_FILE}")

    # debugging
    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


if __name__ == "__main__":
    main_demo()
