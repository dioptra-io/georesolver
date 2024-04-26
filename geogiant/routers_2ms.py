"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio

from typing import Generator
from ipaddress import IPv4Address, AddressValueError
from pathlib import Path
from loguru import logger

from geogiant.common.files_utils import load_csv, dump_json
from geogiant.common.ip_addresses_utils import (
    get_prefix_from_ip,
)
from geogiant.hostname_init import resolve_vps_subnet
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def load_json_iter(file_path: Path) -> Generator:
    """iter load json file"""
    with file_path.open("r") as f:
        for row in f.readlines():
            yield json.loads(row)


def generate_routers_subnet_file() -> None:
    if not (path_settings.DATASET / "routers_subnet.json").exists():
        subnets = set()
        rows = load_json_iter(path_settings.DATASET / "routers_2ms.json")
        for row in rows:
            addr = row["ip"]
            # remove IPv6 and private IP addresses
            try:
                if not IPv4Address(addr).is_private:
                    subnet = get_prefix_from_ip(addr)
                    subnets.add(subnet)
            except AddressValueError:
                continue

        logger.info(f"Number of subnets in routers datasets:: {len(subnets)}")
        subnets = list(subnets)

        dump_json(subnets, path_settings.DATASET / "routers_subnet.json")


async def main() -> None:
    """init main"""

    selected_hostnames = load_csv(
        path_settings.DATASET / "selected_hostname_geo_score.csv"
    )

    input_file = path_settings.DATASET / "routers_subnet.json"
    output_file = path_settings.RESULTS_PATH / "routers_ecs_resolution.csv"

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=input_file,
        output_file=output_file,
        chunk_size=100,
    )


if __name__ == "__main__":
    asyncio.run(main())
