"""run GeoResolver geolocation on a set of demo targets"""

from georesolver.georesolver import run_georesolver
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/itdk_config.json"


def main() -> None:
    """load config and start experiment"""
    run_georesolver(CONFIG_PATH)


if __name__ == "__main__":
    main()
