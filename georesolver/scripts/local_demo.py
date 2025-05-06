"""run GeoResolver geolocation on a set of demo targets"""

from random import sample

from georesolver.agent.main import main
from georesolver.common.files_utils import load_csv, dump_csv, dump_json
from georesolver.common.settings import PathSettings, RIPEAtlasSettings

path_settings = PathSettings()
ripe_atlas_settings = RIPEAtlasSettings()

UUID = "59cd1877-cd58-4ff9-ad7f-41fa8ad26a3f"
NB_ADDRS = 100
USER_TARGET_FILE = path_settings.USER_DATASETS / "demo_targets.csv"
LOCAL_DEMO_TARGET_FILE = path_settings.DATASET / "itdk/random_sample_private_db.csv"
LOCAL_DEMO_HOSTNAME_FILE = path_settings.USER_DATASETS / "hostnames_georesolver.csv"
ECS_TABLE: str = "georesolver_itdk_sample_ecs"
SCORE_TABLE: str = "georesolver_itdk_sample_score"
SCHEDULE_TABLE: str = "georesolver_itdk_sample_schedule"
PING_TABLE: str = "georesolver_itdk_sample_ping"
GEOLOC_TABLE: str = "georesolver_itdk_sample_geoloc"
BATCH_SIZE = 1_000


def generate_random_dataset() -> None:
    """generate a random target IP addresses, output into target file"""
    ip_addrs = load_csv(USER_TARGET_FILE)
    sample_targets = sample(ip_addrs, NB_ADDRS)
    dump_csv(sample_targets, LOCAL_DEMO_TARGET_FILE)


if __name__ == "__main__":

    # generate_random_dataset()

    agent_config = {
        "agent_uuid": "6d4b3d4f-9b22-41e1-8f00-d556a1a107f9",
        "experiment_name": "georesolver_itdk_sample",
        "experiment_uuid": "59cd1877-cd58-4ff9-ad7f-41fa8ad26a3f",
        "target_file": f"{LOCAL_DEMO_TARGET_FILE.resolve()}",
        "hostname_file": f"{LOCAL_DEMO_HOSTNAME_FILE.resolve()}",
        "batch_size": BATCH_SIZE,
        "init_ecs_mapping": False,
        "log_path": f"{(path_settings.LOG_PATH / 'georesolver_itdk_sample/').resolve()}",
        "processes": [
            {
                "name": "ecs_process",
                "in_table": ECS_TABLE,
                "out_table": ECS_TABLE,
            },
            {
                "name": "score_process",
                "in_table": ECS_TABLE,
                "out_table": SCORE_TABLE,
            },
            {
                "name": "schedule_process",
                "in_table": SCORE_TABLE,
                "out_table": SCHEDULE_TABLE,
            },
            {
                "name": "ping_process",
                "in_table": SCORE_TABLE,
                "out_table": PING_TABLE,
            },
            {
                "name": "insert_process",
                "in_table": PING_TABLE,
                "out_table": GEOLOC_TABLE,
            },
        ],
        "agents": [
            {
                "user": "hugo",
                "host": "localhost",
                "remote_dir": "/srv/hugo/georesolver",
                "agent_uuid": "6d4b3d4f-9b22-41e1-8e00-d556a1a107f9",
                "agent_processes": ["ecs_process", "score_process"],
            }
        ],
    }

    dump_json(
        agent_config, path_settings.LOG_PATH / "local_demo/local_demo_config.json"
    )

    main(
        path_settings.LOG_PATH / "local_demo/local_demo_config.json",
        check_cached_measurements=False,
    )
