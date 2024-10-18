"""this script is meant to run periodically to test if the VPs' ECS resolution becomes stale throught time"""

from uuid import uuid4
from pathlib import Path
from loguru import logger
from datetime import datetime

from geogiant.scheduler import create_agents
from geogiant.agent import run_dns_mapping, ProcessNames
from geogiant.clickhouse.queries import load_targets, load_vps, create_tmp_csv_file
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.files_utils import load_json, dump_json, dump_csv
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

CONFIG_PATH = path_settings.DEFAULT / "../experiment_config/config_local_example.json"
TARGET_FILE = path_settings.DATASET / "ripe_atlas_anchors_targets.csv"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.csv"
HOSTNAME_FILE = path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv"
DEMO_ECS_MAPPING_TABLE = "demo_ecs_mapping"
NB_ADDRS = 10_000


def update_config_uuids(config_path: Path) -> str:
    """update the agent uuid of experiment config (to have conflict with older measurements)"""
    new_experiment_uuid = uuid4()
    new_agent_uuid = uuid4()
    config = load_json(config_path)

    # save previous uuids in case of failure
    # set new agent uuid
    config["experiment_uuid"] = str(new_experiment_uuid)
    config["agents"][0]["agent_uuid"] = str(new_agent_uuid)

    # dump updated config
    dump_json(config, config_path)

    return str(new_experiment_uuid)


def update_input_vps_ecs_table(
    config_path: Path,
    vps_ecs_table: str,
) -> None:
    """update score process in config to use the right VPs table"""
    config = load_json(config_path)

    for process, process_def in config["processes"].items():
        if process == ProcessNames.SCORE_PROC.value:
            process_def["vps_ecs_table"] = vps_ecs_table

    dump_json(config, config_path)


def main_ecs_deprecation() -> None:
    """run VPs ECS deprecation experiment"""
    # 1. load targets (RIPE Atlas anchors)
    # 2. day 1: perform DNS resolution and georesolver on targets
    # 3. following days:
    #       - perform new DNS resolution
    #       - perform geolocation with new resolution
    #       - perform geolocation with day 1 resolution
    # dump N random address in demo file

    # generate ripe atlas anchors and vps subnet dataset
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    targets = targets[:2]
    dump_csv(targets, TARGET_FILE)

    vps = load_vps(clickhouse_settings.VPS_FILTERED_FINAL_TABLE)
    vps_subnet = [get_prefix_from_ip(v["subnet_v4"]) for v in vps][:3]
    dump_csv(vps_subnet, VPS_SUBNET_PATH)

    # update config
    update_config_uuids()

    # run ecs on vps
    run_dns_mapping(
        subnets=vps_subnet,
        hostname_file=HOSTNAME_FILE,
        output_table="ecs_deprecation__init_vps_ecs",
    )

    # update config so score process use the right vps ecs table
    update_input_vps_ecs_table(CONFIG_PATH, "ecs_deprecation__init_vps_ecs")

    # run georesolver
    agents = create_agents(CONFIG_PATH)

    for agent in agents:
        agent.run()


if __name__ == "__main__":
    main_ecs_deprecation()
