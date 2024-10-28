"""Evaluate if the ECS resolution are persistent throught time"""

import asyncio

from pathlib import Path
from datetime import datetime

from georesolver.agent import run_dns_mapping
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def repeat_resolusiont(
    subnets: list,
    hostname_file: Path,
    repeat: bool = False,
    chunk_size: int = 500,
    output_file: Path = None,
    output_table: str = None,
    end_date: datetime = "2024-01-23 00:00",
    waiting_time: int = 60,
    request_timout: float = 0.1,
    request_type: str = "A",
    output_logs: Path = None,
) -> None:
    """repeat zdns measurement on set of VPs"""

    await run_dns_mapping(
        subnets=subnets,
        hostname_file=hostname_file,
        chunk_size=chunk_size,
        output_file=output_file,
        output_table=output_table,
        request_timout=request_timout,
        request_type=request_type,
        output_logs=output_logs,
    )

    if repeat:
        await asyncio.sleep(waiting_time)  # wait two hours
        while datetime.today() < end_date:
            await run_dns_mapping(
                subnets=subnets,
                hostname_file=hostname_file,
                chunk_size=chunk_size,
                output_file=output_file,
                output_table=output_table,
                request_timout=request_timout,
                request_type=request_type,
                output_logs=output_logs,
            )
            await asyncio.sleep(waiting_time)
