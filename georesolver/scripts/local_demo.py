"""run GeoResolver geolocation on a set of demo targets"""

import os
from loguru import logger

from georesolver.main import main
from georesolver.evaluation.itdk_dataset import get_random_itdk_routers
from georesolver.common.files_utils import dump_json
from georesolver.common.settings import PathSettings, RIPEAtlasSettings

path_settings = PathSettings()
ripe_atlas_settings = RIPEAtlasSettings()

# override api key if needed, comment if not
os.environ["RIPE_ATLAS_SECRET_KEY"] = (
    RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY_SECONDARY
)


UUID = "59cd1877-cd58-4ff9-ad7f-41fa8ad26a3f"
# LOCAL_DEMO_TARGET_FILE = path_settings.DATASET / "local_demo_targets.csv"
# LOCAL_DEMO_TARGET_FILE = path_settings.DATASET / "ripe_atlas_itdktargets.csv"
# LOCAL_DEMO_TARGET_FILE = path_settings.DATASET / "itdk/itdk_targets.csv"
LOCAL_DEMO_TARGET_FILE = (
    path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv"
)
LOCAL_DEMO_HOSTNAME_FILE = path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv"
LOCAL_DEMO_CONFIG_PATH = path_settings.LOG_PATH / "local_demo"
LOCAL_DEMO_ECS_MAPPING_TABLE = "local_demo_ecs_mapping"
LOCAL_DEMO_SCORE_TABLE = "local_demo_score"
LOCAL_DEMO_PING_TABLE = "local_demo_ping"
LOCAL_DEMO_GEOLOC_TABLE = "local_demo_geoloc"
NB_ADDRS = 100
BATCH_SIZE = 1_000

DOCKER_EXEC = True


def local_exec() -> None:
    """run georesolver locally"""


if __name__ == "__main__":
    # dump N random address in local_demo file
    # logger.info(f"Generating {NB_ADDRS} random target file for local_demo")
    # random_router_interfaces = get_random_itdk_routers(
    #     NB_ADDRS,
    #     LOCAL_DEMO_TARGET_FILE,
    #     mode="w",
    # )

    logger.info(f"File available at:: {LOCAL_DEMO_TARGET_FILE}")

    # create agent config
    agent_config = {
        "agent_uuid": "16b489a-5191-459c-9f4e-da558d58836c",
        "target_file": f"{LOCAL_DEMO_TARGET_FILE.resolve()}",
        "hostname_file": f"{LOCAL_DEMO_HOSTNAME_FILE.resolve()}",
        "batch_size": BATCH_SIZE,
        "init_ecs_mapping": False,
        "processes": [
            {
                "name": "ecs_process",
                "in_table": "itdk_ecs",
                "out_table": "itdk_ecs",
            },
            {
                "name": "score_process",
                "in_table": "itdk_ecs",
                "out_table": "itdk_score",
            },
            {
                "name": "ping_process",
                "in_table": "itdk_score",
                "out_table": "itdk_ping",
            },
            {
                "name": "insert_process",
                "in_table": "itdk_ping",
                "out_table": "itdk_geoloc",
            },
        ],
        "log_path": f"{LOCAL_DEMO_CONFIG_PATH.resolve()}",
    }

    dump_json(
        agent_config, path_settings.LOG_PATH / "local_demo/local_demo_config.json"
    )

    main(
        path_settings.LOG_PATH / "local_demo/local_demo_config.json",
        check_cached_measurements=True,
    )
