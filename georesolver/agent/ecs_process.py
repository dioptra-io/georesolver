from pathlib import Path
from datetime import datetime
from loguru import logger

from georesolver.zdns import ZDNS
from georesolver.clickhouse.queries import get_subnets
from georesolver.common.files_utils import load_csv, create_tmp_csv_file
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()


async def run_dns_mapping(
    subnets: list,
    hostname_file: Path,
    chunk_size: int = 500,
    name_servers="8.8.8.8",
    output_file: Path = None,
    output_table: str = None,
    request_timout: float = 0.1,
    request_type: str = "A",
    output_logs: Path = None,
    itterative: bool = False,
    ipv6: bool = False,
) -> None:
    """perform DNS mapping with zdns on VPs subnet"""

    subnets = [subnet + "/24" if not ipv6 else subnet + "/24" for subnet in subnets]

    with hostname_file.open("r") as f:
        all_hostnames = f.readlines()

        logger.info(
            f"Resolving:: {len(all_hostnames)} hostnames on {len(subnets)} subnets"
        )

        # split file
        for index in range(0, len(all_hostnames), chunk_size):
            hostnames = all_hostnames[index : index + chunk_size]

            tmp_hostname_file = create_tmp_csv_file(hostnames)

            logger.info(
                f"Starting to resolve hostnames {index} to {index + chunk_size} (total={len(all_hostnames)})"
            )

            zdns = ZDNS(
                subnets=subnets,
                hostname_file=tmp_hostname_file,
                output_file=output_file,
                output_table=output_table,
                name_servers=name_servers,
                timeout=request_timout,
                request_type=request_type,
                output_logs=output_logs,
                iterative=itterative,
            )
            await zdns.main()

            tmp_hostname_file.unlink()


async def filter_ecs_subnets(subnets: list[str], ecs_mapping_table: str) -> list[str]:
    """retrieve all subnets for which ECS resolution was done"""
    cached_ecs_subnets = get_subnets(table_name=ecs_mapping_table)

    filtered_subnets = set(subnets).difference(set(cached_ecs_subnets))

    return list(filtered_subnets)


async def ecs_task(
    target_file: Path,
    hostname_file: Path,
    in_table: str,
    out_table: str,
    batch_size: int = 1_000,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "ecs_task.log",
    dry_run: bool = False,
) -> None:
    """run ecs mapping on batches of target subnets"""
    setup_logger(log_path / output_logs)

    targets = load_csv(target_file, exit_on_failure=True)
    subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))

    # Check ECS mapping exists
    filtered_subnets = await filter_ecs_subnets(subnets, in_table)
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

            await run_dns_mapping(
                subnets=input_subnets,
                hostname_file=hostname_file,
                output_table=out_table,
                output_logs=log_path / output_logs,
            )
    else:
        logger.info("Skipping ECS resolution because all subnet mapping is done")
