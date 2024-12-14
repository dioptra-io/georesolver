"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger
from pprint import pformat

from georesolver.main import main
from georesolver.scheduler import create_agents
from georesolver.common.files_utils import load_json
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/itdk_config.json"


def main() -> None:
    """load config and start experiment"""
    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


if __name__ == "__main__":
    main()
