"""run GeoResolver ECS and score process on the entire IPv4 address space that answer to pings"""

from georesolver.scheduler import create_agents
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/internet_scale_config.json"


def main() -> None:
    """run georesolver on a random subset of ITDK IP addresses"""
    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


if __name__ == "__main__":
    main()
