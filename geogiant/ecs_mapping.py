from loguru import logger

from geogiant.common.files_utils import (
    create_tmp_json_file,
)
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def ecs_mapping(subnets: list[str], hostnames: list[str], ecs_table: str) -> None:
    batch_size = 10_000
    for i in range(0, len(subnets), batch_size):
        logger.info(f"Batch:: {i+1}/{len(subnets) // batch_size}")
        batch_subnets = subnets[i : i + batch_size]

        subnet_tmp_file_path = create_tmp_json_file(batch_subnets)

        await resolve_vps_subnet(
            selected_hostnames=hostnames,
            input_file=subnet_tmp_file_path,
            output_table=ecs_table,
            chunk_size=500,
        )

        subnet_tmp_file_path.unlink()
