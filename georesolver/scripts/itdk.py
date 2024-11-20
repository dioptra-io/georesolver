"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger
from pprint import pformat

from georesolver.main import main
from georesolver.scheduler import create_agents
from georesolver.common.files_utils import load_json
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/itdk_config.json"


def run_experiment() -> None:
    """load config and start experiment"""
    config = load_json(CONFIG_PATH)
    logger.info("Running subnet aggregation experiment from config:: ")
    logger.info("config=\n{}", pformat(config))

    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


def main() -> None:
    """
    measurement entry point:
        - generate target file:
            - takes N /24 subnets with at least 25% responsive IP addresses
            - takes /24 subnets that belongs to larger prefixes (in MaxMind dataset)
        - run georesolver measurement
    """
    run_experiment()


if __name__ == "__main__":
    main()
