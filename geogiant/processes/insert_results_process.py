import asyncio

from pathlib import Path
from loguru import logger

from geogiant.prober import RIPEAtlasProber
from geogiant.common.queries import load_cached_targets
from geogiant.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def insert_results_task(
    targets: list[str],
    ping_table: str,
    geoloc_table: str,
    measurement_uuid: str,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "insert_results_task.log",
    dry_run: bool = False,
) -> None:
    setup_logger(log_path / output_logs)

    # remove targets for which we already made measurements
    cached_targets = load_cached_targets(ping_table)

    # remove targets for which a measurement was started but results not inserted yet
    nb_targets = len(set(targets).difference(set(cached_targets)))

    if nb_targets == 0:
        logger.info("All measurements done, insert results process stopped")
        return
    else:
        logger.info(f"Remainning targets to geolocate:: {nb_targets}")

    if dry_run:
        logger.info("Stopped insert results task")

    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=measurement_uuid,
        output_table=ping_table,
        output_logs=log_path / output_logs,
    )

    await prober.insert_results(nb_targets, geoloc_table)

    logger.info(f"All pings and geoloc inserted, measurement finished")
