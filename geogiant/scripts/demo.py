"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger

from geogiant.scheduler import create_agents
from geogiant.evaluation.itdk_dataset import get_random_itdk_routers
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

DEMO_TARGET_FILE = path_settings.DATASET / "demo_targets.csv"
DEMO_ECS_MAPPING_TABLE = "demo_ecs_mapping"
DEMO_SCORE_TABLE = "demo_score"
DEMO_PING_TABLE = "demo_ping"
DEMO_GEOLOC_TABLE = "demo_geoloc"
NB_ADDRS = 100


def main_demo() -> None:
    """run georesolver on a random subset of ITDK IP addresses"""
    # dump N random address in demo file
    logger.info(f"Generating {NB_ADDRS} random target file for demo")
    get_random_itdk_routers(
        NB_ADDRS,
        DEMO_TARGET_FILE,
        mode="w",
    )
    logger.info(f"File available at:: {DEMO_TARGET_FILE}")

    # debugging
    config_path = (
        path_settings.DEFAULT / "../experiment_config/config_local_example.json"
    )
    agents = create_agents(config_path)
    for agent in agents:
        agent.run()


if __name__ == "__main__":
    main_demo()
