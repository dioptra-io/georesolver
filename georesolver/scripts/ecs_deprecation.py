"""this script is meant to run periodically to test if the VPs' ECS resolution becomes stale throught time"""

import os
import sys
import asyncio

from uuid import uuid4
from pathlib import Path
from loguru import logger
from pprint import pformat

from georesolver.scheduler import create_agents
from georesolver.agent import run_dns_mapping, ProcessNames
from georesolver.clickhouse.queries import load_targets, load_vps, get_tables
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.files_utils import load_json, dump_json, dump_csv
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
# overwrite clickhouse settings
os.environ["CLICKHOUSE_DATABASE"] = "GeoResolver_ecs_deprecation"
# override api key if needed, comment if not
os.environ["RIPE_ATLAS_SECRET_KEY"] = (
    RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY_SECONDARY
)

# input files paths
EXPERIMENT_NAME = "ecs_deprecation"
CONFIG_PATH = path_settings.DATASET / "experiment_config/ecs_deprecation_config.json"
TARGET_FILE = path_settings.DATASET / "ripe_atlas_anchors_targets.csv"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.csv"
HOSTNAME_FILE = path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv"

# clickhouse tables experiments
VPS_INIT_ECS_TABLE = EXPERIMENT_NAME + "__vps_init_ecs"

# comment if debug needed
logger.remove()
logger.add(sys.stdout, level="INFO")


def print_header(msg: str) -> None:
    """log header like message"""
    logger.info("########################################")
    logger.info(f"# {msg}")
    logger.info("########################################")


def parse_uuid(uuid: str) -> str:
    """clickhouse only allow underscore for name tables"""
    return uuid.replace("-", "_")


def load_datasets(target_file: Path, vps_subnet_file: Path) -> None:
    """load ripe atlas anchors, vps subnets and output files in defined dirs"""
    # generate ripe atlas anchors and vps subnet dataset
    targets = load_targets(ch_settings.VPS_FILTERED_FINAL_TABLE)
    targets = [target["addr"] for target in targets]
    dump_csv(targets, target_file)

    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    vps_subnet = [get_prefix_from_ip(v["subnet"]) for v in vps]
    dump_csv(vps_subnet, vps_subnet_file)

    return targets, vps_subnet


def update_config_uuids(config: dict) -> str:
    """update the agent uuid of experiment config (to have conflict with older measurements)"""
    config["experiment_uuid"] = str(uuid4())
    config["agents"][0]["agent_uuid"] = str(uuid4())

    return config


def update_in_vps_ecs_table(config: dict, vps_ecs_table: str) -> None:
    """update score process in config to use the right VPs table"""
    for process_def in config["processes"]:
        if process_def["name"] == ProcessNames.SCORE_PROC.value:
            process_def["vps_ecs_table"] = vps_ecs_table

    return config


def update_out_target_tables(config: dict) -> None:
    """update score process in config to use the right VPs table"""

    for process_def in config["processes"]:
        if process_def["name"] == ProcessNames.ECS_PROC.value:
            prefix_in = "ecs"
            prefix_out = "ecs"
        if process_def["name"] == ProcessNames.SCORE_PROC.value:
            prefix_in = "ecs"
            prefix_out = "score"
        if process_def["name"] == ProcessNames.PING_PROC.value:
            prefix_in = "score"
            prefix_out = "ping"
        if process_def["name"] == ProcessNames.INSERT_PROC.value:
            prefix_in = "ping"
            prefix_out = "geoloc"

        process_def["in_table"] = (
            EXPERIMENT_NAME + f"__{parse_uuid(config['experiment_uuid'])}_{prefix_in}"
        )
        process_def["out_table"] = (
            EXPERIMENT_NAME + f"__{parse_uuid(config['experiment_uuid'])}_{prefix_out}"
        )

    return config


def update_config(
    experiment_name: str,
    config_path: Path,
    vps_ecs_table: str = None,
) -> dict:
    """
    update the experiment config (uuids, vps ecs table and out tables) between each round
    override is ok because scheduler output newly created config to experiement path
    """
    config = load_json(config_path)
    config["target_file"] = str(TARGET_FILE.resolve())

    # update config
    config = update_config_uuids(config)
    new_experiment_uuid = config["experiment_uuid"]

    # create new vps ecs table with experiment uuid
    if not vps_ecs_table:
        vps_ecs_table = experiment_name + f"__{parse_uuid(new_experiment_uuid)}_vps_ecs"

    # update config so score process use the right vps ecs table
    config = update_in_vps_ecs_table(config, vps_ecs_table)

    # update target output table
    config = update_out_target_tables(config)

    dump_json(config, config_path)

    return config


def table_exists(table_name: str) -> bool:
    """check if a table exists"""
    tables = get_tables()

    if table_name in tables:
        return True

    return False


async def init_experiment(vps_subnet: list[str]) -> None:
    """step 0: perform ECS with vps and run georesolver"""
    # create fresh experiment config
    update_config(
        experiment_name=EXPERIMENT_NAME,
        config_path=CONFIG_PATH,
        vps_ecs_table=VPS_INIT_ECS_TABLE,
    )

    # run ECS mapping resolution init
    await run_dns_mapping(
        subnets=vps_subnet,
        hostname_file=HOSTNAME_FILE,
        output_table=VPS_INIT_ECS_TABLE,
    )

    # run georesolver
    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


async def run_experiment(vps_subnet: list[str]) -> None:
    """step 1: run the experiment: 1) ecs vps mapping, 2) georesolver old, 3) georesolver new"""

    print_header("GeoResolver:: old VPs mapping")
    logger.info(f"Running Georesolver with VPs ECS mapping:: {VPS_INIT_ECS_TABLE}")

    # create fresh experiment config
    update_config(
        experiment_name=EXPERIMENT_NAME,
        config_path=CONFIG_PATH,
        vps_ecs_table=VPS_INIT_ECS_TABLE,
    )

    # run georesolver
    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()

    new_config = update_config(
        experiment_name=EXPERIMENT_NAME,
        config_path=CONFIG_PATH,
    )

    print_header("GeoResolver:: new VPs mapping")
    logger.debug("Running new experiment round with config::\n{}", pformat(new_config))

    # extract vps ecs table from newly created config
    vps_ecs_table = [
        proc_def["vps_ecs_table"]
        for proc_def in new_config["processes"]
        if proc_def["name"] == ProcessNames.SCORE_PROC.value
    ][0]

    logger.info(f"Running Georesolver with new VPs ECS mapping:: {vps_ecs_table}")

    # run ecs on vps
    await run_dns_mapping(
        subnets=vps_subnet,
        hostname_file=HOSTNAME_FILE,
        output_table=vps_ecs_table,
    )

    # run georesolver
    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


async def main() -> None:
    """
    run VPs ECS deprecation experiment
    1. load targets (RIPE Atlas anchors)
    2. init   : perform DNS resolution and georesolver on targets
    3. rounds :
          - perform new DNS resolution
          - perform geolocation with new resolution
          - perform geolocation with day 1 resolution
    """

    targets, vps_subnet = load_datasets(TARGET_FILE, VPS_SUBNET_PATH)

    if not table_exists(VPS_INIT_ECS_TABLE):
        print_header("ECS deprecation:: init experiment")
        logger.info(f"Nb targets     :: {len(targets)}")
        logger.info(f"Nb VPs subnets :: {len(vps_subnet)}")
        await init_experiment(vps_subnet)
    else:
        print_header("ECS deprecation:: running experiment round")
        logger.info(f"Nb targets     :: {len(targets)}")
        logger.info(f"Nb VPs subnets :: {len(vps_subnet)}")
        await run_experiment(vps_subnet)


if __name__ == "__main__":
    asyncio.run(main())
