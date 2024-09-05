import asyncio

from uuid import uuid4
from loguru import logger
from collections import defaultdict
from datetime import datetime, timedelta

from geogiant.common.files_utils import load_csv, load_json
from geogiant.ecs_mapping_init import resolve_vps_subnet
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def load_hostnames() -> dict:
    selected_hostnames_per_cdn_per_ns = load_json(
        path_settings.DATASET
        / f"hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json"
    )

    selected_hostnames = set()
    selected_hostnames_per_cdn = defaultdict(list)
    for ns in selected_hostnames_per_cdn_per_ns:
        for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
            selected_hostnames.update(hostnames)
            selected_hostnames_per_cdn[org].extend(hostnames)

    logger.info(f"{len(selected_hostnames)=}")

    return selected_hostnames_per_cdn, selected_hostnames


async def main() -> None:
    """init main"""
    input_file = path_settings.DATASET / "vps_subnet.json"
    _, selected_hostnames = load_hostnames()

    end_date = datetime.today() + timedelta(days=30)

    logger.info(f"ECS resolution on VPs subnets")
    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=input_file,
        output_table="persistency_analysis_latest",
        chunk_size=500,
        repeat=True,
        waiting_time=0,
        end_date=end_date,
    )


if __name__ == "__main__":
    asyncio.run(main())
