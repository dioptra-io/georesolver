"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger

from geogiant.main import main
from geogiant.common.files_utils import load_csv, dump_csv
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

ITDK_RESPONSIVE_ROUTER_INTERFACE = (
    path_settings.DATASET / "itdk/itdk_responsive_router_interface.csv"
)
ITDK_TARGET_FILE = path_settings.DATASET / "itdk/itdk_target_file.csv"
ITDK_ECS_MAPPING_TABLE = "itdk_ecs_mapping"
ITDK_SCORE_TABLE = "itdk_score"
ITDK_PING_TABLE = "itdk_ping"
ITDK_GEOLOC_TABLE = "itdk_geoloc"

if __name__ == "__main__":
    logger.info(f"Starting ITDK measurements::")
    logger.info(f"Target file:: {ITDK_TARGET_FILE}")

    if not ITDK_TARGET_FILE.exists():
        itdk_targets = load_csv(ITDK_RESPONSIVE_ROUTER_INTERFACE)
        itdk_targets = [row.split(",")[-1] for row in itdk_targets]
        dump_csv(itdk_targets, ITDK_TARGET_FILE)

    main(
        target_file=ITDK_TARGET_FILE,
        hostname_file=path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv",
        ecs_mapping_table=ITDK_ECS_MAPPING_TABLE,
        score_table=ITDK_SCORE_TABLE,
        ping_table=ITDK_PING_TABLE,
        geoloc_table=ITDK_GEOLOC_TABLE,
        log_path=path_settings.LOG_PATH / "itdk",
        batch_size=1_000,
    )
