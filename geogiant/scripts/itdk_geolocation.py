"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger

from geogiant.main import main
from geogiant.common.files_utils import load_csv, create_tmp_csv_file
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

ITDK_TARGET_FILE = path_settings.DATASET / "itdk/itdk_responsive_router_interface.csv"
ITDK_ECS_MAPPING_TABLE = "itdk_ecs_mapping"
ITDK_SCORE_TABLE = "itdk_score"
ITDK_PING_TABLE = "itdk_ping"
TARGETS_PER_MEASUREMENT = 10_000

if __name__ == "__main__":
    logger.info(f"Starting ITDK measurements::")
    logger.info(f"Target file:: {ITDK_TARGET_FILE}")

    itdk_targets = load_csv(ITDK_TARGET_FILE)
    itdk_targets = [row.split(",")[-1] for row in itdk_targets]

    logger.info(f"Total number of targets:: {len(itdk_targets)}")
    logger.info(f"Running on batches of:: {TARGETS_PER_MEASUREMENT}")

    for i in range(0, len(itdk_targets), TARGETS_PER_MEASUREMENT):

        batch_targets = itdk_targets[i : i + TARGETS_PER_MEASUREMENT]

        logger.info(
            f"Running batch:: {(i//TARGETS_PER_MEASUREMENT)+1}/{(len(itdk_targets) // TARGETS_PER_MEASUREMENT) +1}"
        )

        tmp_target_file = create_tmp_csv_file(batch_targets)

        main(
            target_file=tmp_target_file,
            hostname_file=path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv",
            ecs_mapping_table=ITDK_ECS_MAPPING_TABLE,
            score_table=ITDK_SCORE_TABLE,
            ping_table=ITDK_PING_TABLE,
            log_path=path_settings.LOG_PATH / "itdk",
            batch_size=1_000,
        )

        tmp_target_file.unlink()

        break
