"""central script for running georesolver evaluation CoNEXT 2025 evaluation"""

from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


def main() -> None:
    """entry point, set booleans to True or False to either run evaluation or not"""
    do_tier1_evaluation: bool = True


if __name__ == "__main__":
    main()
