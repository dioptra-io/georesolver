from pathlib import Path
from loguru import logger

from geogiant.ecs_mapping_init import resolve_hostnames
from geogiant.common.queries import get_subnets_mapping
from geogiant.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()


async def filter_ecs_subnets(subnets: list[str], ecs_mapping_table: str) -> list[str]:
    """retrieve all subnets for which ECS resolution was done"""
    subnet_mapping = []
    subnet_mapping = get_subnets_mapping(dns_table=ecs_mapping_table, subnets=subnets)

    cached_subnets = [subnet for subnet in subnet_mapping]
    filtered_subnets = set(subnets).difference(set(cached_subnets))

    return list(filtered_subnets)


async def ecs_mapping_task(
    subnets: list[str],
    hostname_file: list[str],
    ecs_mapping_table: str,
    batch_size: int = 1_000,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "ecs_mapping_task.log",
) -> None:
    """run ecs mapping on batches of target subnets"""
    setup_logger(log_path / output_logs)

    # Check ECS mapping exists
    filtered_subnets = await filter_ecs_subnets(subnets, ecs_mapping_table)
    logger.info(f"Original number of subnets:: {len(subnets)}")
    logger.info(f"Remaining subnets to geolocate:: {len(filtered_subnets)}")

    # ECS mapping
    if filtered_subnets:
        for i in range(0, len(filtered_subnets), batch_size):
            input_subnets = filtered_subnets[i : i + batch_size]
            logger.info(
                f"ECS mapping:: batch={(i+1) // batch_size}/{(len(filtered_subnets) // batch_size)}"
            )

            await resolve_hostnames(
                subnets=input_subnets,
                hostname_file=hostname_file,
                output_table=ecs_mapping_table,
                output_logs=log_path / output_logs,
            )
    else:
        logger.info("Skipping ECS resolution because all subnet mapping is done")
