from pathlib import Path
from loguru import logger

from geogiant.ecs_mapping_init import resolve_hostnames
from geogiant.common.queries import get_subnets
from geogiant.common.files_utils import load_csv
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()


def docker_run_cmd() -> str:
    return """
        docker run -d \
        -v "$(pwd)/results:/app/geogiant/results" \
        -v "$(pwd)/datasets/all_ecs_selected_hostnames.csv:/app/geogiant/datasets/all_ecs_selected_hostnames.csv" \
        -v "$(pwd)/datasets/vm_config.json:/app/geogiant/datasets/vm_config.json" \
        --network host \
        ghcr.io/dioptra-io/geogiant:main
    """


async def filter_ecs_subnets(subnets: list[str], ecs_mapping_table: str) -> list[str]:
    """retrieve all subnets for which ECS resolution was done"""
    cached_ecs_subnets = get_subnets(table_name=ecs_mapping_table)

    filtered_subnets = set(subnets).difference(set(cached_ecs_subnets))

    return list(filtered_subnets)


async def ecs_mapping_agents(
    agents: tuple[str],
    measurement_config: dict,
) -> None:
    """TODO:
    for each server:
        - distribute load by assigning a set of subnets to each agent
        - create instance of ecs mapping task with docker (if none on the server)
        - upload target file to distant server
        - start ecs mapping task
        inser results:
            a) insert remotely into Clickhouse
            b) output file, copy from distant, insert locally
    """
    # create ecs agents
    pass


async def ecs_task(
    target_file: Path,
    hostname_file: Path,
    ecs_mapping_table: str,
    batch_size: int = 1_000,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "ecs_mapping_task.log",
    dry_run: bool = False,
) -> None:
    """run ecs mapping on batches of target subnets"""
    setup_logger(log_path / output_logs)

    targets = load_csv(target_file, exit_on_failure=True)
    subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))

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

            if dry_run:
                logger.info("Stopped ECS mapping process")
                break

            await resolve_hostnames(
                subnets=input_subnets,
                hostname_file=hostname_file,
                output_table=ecs_mapping_table,
                output_logs=log_path / output_logs,
            )
    else:
        logger.info("Skipping ECS resolution because all subnet mapping is done")
