import asyncio

from geogiant.common.files_utils import load_csv
from geogiant.hostname_init import resolve_vps_subnet
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

subnet_file = path_settings.DATASET / "end_to_end_subnets.json"
hostname_file = path_settings.DATASET / "selected_hostnames.csv"
output_file = path_settings.RESULTS_PATH / "end_to_end_ecs_resolution.csv"


async def main() -> None:
    """init main"""

    selected_hostnames = load_csv(hostname_file)

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=subnet_file,
        output_file=output_file,
        chunk_size=100,
    )


if __name__ == "__main__":
    asyncio.run(main())
