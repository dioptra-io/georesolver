"""run GeoResolver geolocation on a set of demo targets"""

from geogiant.main import main
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

DEMO_TARGET_FILE = path_settings.DATASET / "demo_targets.csv"
DEMO_ECS_MAPPING_TABLE = "demo_ecs_mapping"
DEMO_SCORE_TABLE = "demo_score"
DEMO_PING_TABLE = "demo_ping"

if __name__ == "__main__":
    measurement_uuid = "f1a1a4d6-aea8-4809-b6bb-1530efdd40bf"
    main(
        target_file=DEMO_TARGET_FILE,
        hostname_file=path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv",
        ecs_mapping_table=DEMO_ECS_MAPPING_TABLE,
        score_table=DEMO_SCORE_TABLE,
        ping_table=DEMO_PING_TABLE,
        log_path=path_settings.LOG_PATH / "demo",
        measurepment_uuid=measurement_uuid,
        batch_size=1_00,
    )
