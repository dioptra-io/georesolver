"""define the main function of a single agent, also entry point of docker image"""

import typer
import asyncio

from loguru import logger
from pathlib import Path
from multiprocessing import Process

from georesolver.agent import ProcessNames
from georesolver.agent.ripe_init import vps_init
from georesolver.common.files_utils import load_csv, load_json
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.agent import (
    ecs_task,
    score_task,
    ping_task,
    insert_task,
    insert_results,
    ecs_init,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def main_processes(task, task_args) -> None:
    """run asynchronous task within new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task(**task_args))


def main(agent_config_path: Path) -> None:

    agent_config = load_json(agent_config_path, exit_on_failure=True)

    # load input path from config
    try:
        agent_uuid = agent_config["agent_uuid"]
        target_file = Path(agent_config["target_file"])
        hostname_file = Path(agent_config["hostname_file"])
        batch_size = agent_config["batch_size"]
        process_definitions = agent_config["processes"]
        log_path = Path(agent_config["log_path"])
        dry_run = agent_config["dry_run"] if "dry_run" in agent_config else False
    except KeyError as e:
        raise RuntimeError(f"Parameter {e} missing in agent configuration")

    # check if input dir exists, exit if not, create logs dir
    if not target_file.exists():
        raise RuntimeError(f"Target file does not exists:: {target_file}")
    if not hostname_file.exists():
        raise RuntimeError(f"Hostname file does not exists:: {hostname_file}")
    if not log_path.exists():
        log_path.mkdir(parents=True)

    # setup logging, both file and stdout
    setup_logger(
        log_path / "main.log",
        verbose=agent_config.get("verbose"),
        to_stdout=True,
    )

    # optional init steps
    if agent_config.get("init_vps"):
        logger.info(f"Starting VPs init (this step might take several days)")
        asyncio.run(vps_init(True, True))

    if agent_config.get("init_ecs_mapping"):
        asyncio.run(ecs_init(hostname_file))

    # load targets, subnets and hostnames
    targets = load_csv(target_file, exit_on_failure=True)
    hostnames = load_csv(hostname_file, exit_on_failure=True)
    subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))

    logger.info("##########################################")
    logger.info(f"# Starting georesolver agent")
    logger.info("##########################################")
    logger.info(f"Experiment uuid     :: {agent_uuid}")
    logger.info(f"Number of targets   :: {len(targets)}")
    logger.info(f"Number of subnets   :: {len(subnets)}")
    logger.info(f"Number of hostnames :: {len(hostnames)}")
    logger.info(f"Batch size          :: {batch_size}")

    logger.info("##########################################")
    logger.info("# Output dirs and table")
    logger.info("##########################################")

    # agent process definition
    processes = []
    for process_definition in process_definitions:
        name = process_definition["name"]
        in_table = process_definition["in_table"]
        out_table = process_definition["out_table"]

        # base set of parameters for each agent's process
        base_params = {
            "target_file": target_file,
            "hostname_file": hostname_file,
            "in_table": in_table,
            "out_table": out_table,
            "agent_uuid": agent_uuid,
            "log_path": log_path,
            "batch_size": batch_size,
            "dry_run": dry_run,
        }

        # get process associated task and modify parameters if needed
        if name == ProcessNames.ECS_PROC.value:
            func = ecs_task
            base_params.pop("agent_uuid")
        if name == ProcessNames.SCORE_PROC.value:
            func = score_task
            base_params.pop("agent_uuid")
            base_params["vps_ecs_table"] = (
                process_definition["vps_ecs_table"]
                if "vps_ecs_table" in process_definition
                else clickhouse_settings.VPS_ECS_MAPPING_TABLE
            )
        if name == ProcessNames.PING_PROC.value:
            func = ping_task
            base_params.pop("hostname_file")
            base_params.pop("batch_size")
        if name == ProcessNames.INSERT_PROC.value:
            func = insert_task
            base_params.pop("hostname_file")

            # check if program crashed, insert measurements if any found
            logger.info("Checking previous measurements insertion (avoid dupplication)")
            asyncio.run(
                insert_results(
                    targets=targets,
                    probing_type="ping",
                    probing_tag=agent_uuid,
                    ping_table=in_table,
                    geoloc_table=out_table,
                    output_logs=log_path / "insert_cache.log",
                    batch_size=batch_size,
                )
            )

        # create process object
        process = Process(target=main_processes, args=(func, base_params))
        processes.append((name, process))

        logger.info(f"Scheduled process {name}:: {in_table=}; {out_table=}")

    # logger.info("##########################################")
    # logger.info("# Starting processes")
    # logger.info("##########################################")
    # process: Process = None
    # for name, process in processes:
    #     # Start all processes
    #     logger.info(f"Starting {name} process")
    #     process.start()

    # logger.info("##########################################")
    # logger.info("# Waiting for process to finish")
    # logger.info("##########################################")
    # process: Process = None
    # for name, process in processes:
    #     # join and wait all processes
    #     process.join()

    # logger.info(f"Agent experiment {agent_uuid} succesfully executed")


if __name__ == "__main__":
    typer.run(main)
