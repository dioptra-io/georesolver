"""run GeoResolver ECS and score process on the entire IPv4 address space that answer to pings"""

from georesolver.georesolver import run_georesolver
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/internet_scale_config.json"


def main() -> None:
    """run georesolver on a random subset of ITDK IP addresses"""
    run_georesolver(CONFIG_PATH)


if __name__ == "__main__":
    main()
