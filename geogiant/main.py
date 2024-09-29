import sys
import typer
import asyncio

from uuid import uuid4
from loguru import logger
from pathlib import Path
from multiprocessing import Process

from geogiant.ripe_init import vps_init
from geogiant.ecs_mapping_init import resolve_hostnames
from geogiant.common.files_utils import load_csv
from geogiant.common.queries import load_vp_subnets
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.processes import (
    ecs_mapping_task,
    score_task,
    geolocation_task,
    insert_results_task,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def main_processes(task, task_args) -> None:
    """run asynchronous task within new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task(**task_args))


def main(
    target_file: Path,
    hostname_file: Path,
    ecs_mapping_table: str = clickhouse_settings.TARGET_ECS_MAPPING_TABLE,
    score_table: str = clickhouse_settings.TARGET_SCORE_TABLE,
    ping_table: str = clickhouse_settings.TARGET_PING_TABLE,
    geoloc_table: str = clickhouse_settings.TARGET_GEOLOC_TABLE,
    log_path: Path = path_settings.LOG_PATH,
    measurement_uuid: str = None,
    batch_size: int = 1_000,
    init_vps: bool = False,
    update_meshed_pings: bool = True,
    update_meshed_traceroutes: bool = True,
    init_ecs_mapping: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    if not log_path.exists():
        log_path.mkdir(parents=True)

    if init_vps:
        logger.info(f"Starting VPs init (this step might take several days)")
        asyncio.run(vps_init(update_meshed_pings, update_meshed_traceroutes))

    if init_ecs_mapping:
        logger.info(
            f"Starting VPs ECS mapping, output table:: {clickhouse_settings.VPS_ECS_MAPPING_TABLE}"
        )

        vps_subnets = load_vp_subnets(clickhouse_settings.VPS_RAW_TABLE)
        asyncio.run(
            resolve_hostnames(
                subnets=vps_subnets,
                hostname_file=hostname_file,
                output_table=clickhouse_settings.VPS_ECS_MAPPING_TABLE,
            )
        )

    # load targets, subnets and hostnames
    targets = load_csv(target_file)
    subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))
    hostnames = load_csv(hostname_file)
    # generate a measurement uuid that defines the entire measurement
    if not measurement_uuid:
        measurement_uuid = str(uuid4())

    # init ECS mapping process
    ecs_mapping_process = Process(
        target=main_processes,
        args=(
            ecs_mapping_task,
            {
                "subnets": subnets,
                "hostname_file": hostname_file,
                "ecs_mapping_table": ecs_mapping_table,
                "log_path": log_path,
                "batch_size": batch_size,
                "dry_run": dry_run,
            },
        ),
    )

    # init Score process
    score_process = Process(
        target=main_processes,
        args=(
            score_task,
            {
                "subnets": subnets,
                "hostnames": hostnames,
                "ecs_mapping_table": ecs_mapping_table,
                "score_table": score_table,
                "log_path": log_path,
                "batch_size": batch_size,
                "dry_run": dry_run,
            },
        ),
    )

    # init Geolocation process
    geolocation_process = Process(
        target=main_processes,
        args=(
            geolocation_task,
            {
                "targets": targets,
                "subnets": subnets,
                "score_table": score_table,
                "ping_table": ping_table,
                "measurement_uuid": measurement_uuid,
                "log_path": log_path,
                "dry_run": dry_run,
            },
        ),
    )

    # init Insert results process
    insert_results_process = Process(
        target=main_processes,
        args=(
            insert_results_task,
            {
                "targets": targets,
                "ping_table": ping_table,
                "geoloc_table": geoloc_table,
                "measurement_uuid": measurement_uuid,
                "log_path": log_path,
                "dry_run": dry_run,
            },
        ),
    )

    logger.info("##########################################")
    logger.info(f"# Starting geolocation")
    logger.info("##########################################")
    logger.info(f"Measurement uuid    :: {measurement_uuid}")
    logger.info(f"Number of targets   :: {len(targets)}")
    logger.info(f"Number of subnets   :: {len(subnets)}")
    logger.info(f"Number of hostnames :: {len(hostnames)}")
    logger.info(f"Batch size          :: {batch_size}")

    logger.info("##########################################")
    logger.info("# Output dirs and table")
    logger.info("##########################################")
    logger.info(f"ECS mapping output table   :: {ecs_mapping_table}")
    logger.info(f"Score output table         :: {score_table}")
    logger.info(f"Ping output table          :: {ping_table}")
    logger.info(f"Geoloc output table        :: {geoloc_table}")
    logger.info(f"Logs output dir            :: {log_path}")

    logger.info("##########################################")
    logger.info("# Starting processes")
    logger.info("##########################################")

    # Start all processes
    logger.info("Starting ECS mapping process")
    ecs_mapping_process.start()

    logger.info("Starting Score process")
    score_process.start()

    logger.info("Starting Geolocation process")
    geolocation_process.start()

    logger.info("Starting Insert results process")
    insert_results_process.start()

    # Wait for each process to finish
    ecs_mapping_process.join()
    score_process.join()
    geolocation_process.join()
    insert_results_process.join()

    logger.info(f"Measurement {measurement_uuid} succesfully done")


if __name__ == "__main__":
    typer.run(main)
