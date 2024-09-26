from pathlib import Path
from loguru import logger

from geogiant.prober import RIPEAtlasProber
from geogiant.common.settings import PathSettings, setup_logger

path_settings = PathSettings()


async def insert_results_task(
    nb_targets: int,
    ping_table: str,
    measurement_uuid: str,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "insert_results_task.log",
    dry_run: bool = False,
) -> None:
    setup_logger(log_path / output_logs)

    if dry_run:
        logger.info("Stopped insert results task")

    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=measurement_uuid,
        output_table=ping_table,
        output_logs=log_path / output_logs,
    )
    await prober.insert_results(nb_targets)
