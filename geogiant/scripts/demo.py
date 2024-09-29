"""run GeoResolver geolocation on a set of demo targets"""

from loguru import logger

from geogiant.main import main
from geogiant.evaluation.itdk_dataset_evaluation import get_random_itdk_routers
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

DEMO_TARGET_FILE = path_settings.DATASET / "demo_targets.csv"
DEMO_ECS_MAPPING_TABLE = "demo_ecs_mapping"
DEMO_SCORE_TABLE = "demo_score"
DEMO_PING_TABLE = "demo_ping"
DEMO_GEOLOC_TABLE = "demo_geoloc"
NB_ADDRS = 1_00

if __name__ == "__main__":
    # dump N random address in demo file
    logger.info(f"Generating {NB_ADDRS} random target file for demo")
    random_router_interfaces = get_random_itdk_routers(NB_ADDRS, DEMO_TARGET_FILE)
    logger.info(f"File available at:: {DEMO_TARGET_FILE}")

    measurement_uuid = "f1a1a4d6-aea8-4809-b6bb-1530efdd40bf"
    main(
        target_file=DEMO_TARGET_FILE,
        hostname_file=path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv",
        ecs_mapping_table=DEMO_ECS_MAPPING_TABLE,
        score_table=DEMO_SCORE_TABLE,
        ping_table=DEMO_PING_TABLE,
        geoloc_table=DEMO_GEOLOC_TABLE,
        log_path=path_settings.LOG_PATH / "demo",
        measurement_uuid=measurement_uuid,
        batch_size=10,
    )
