"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio
import schedule
import time

from typing import Generator
from ipaddress import IPv4Address, AddressValueError
from tqdm import tqdm
from pathlib import Path
from numpy import mean
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.zdns import ZDNS
from geogiant.common.ip_addresses_utils import (
    get_prefix_from_ip,
)
from geogiant.common.files_utils import (
    create_tmp_csv_file,
)
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def resolve_hostnames(
    subnets: list,
    hostname_file: Path,
    output_table: str,
    repeat: bool = False,
    end_date: str = "2024-01-23 00:00",
    chunk_size: int = 100,
) -> None:
    """repeat zdns measurement on set of VPs"""

    async def raw_dns_mapping(subnets: list) -> None:
        """perform DNS mapping with zdns on VPs subnet"""

        subnets = [subnet + "/24" for subnet in subnets]

        with hostname_file.open("r") as f:
            logger.info("raw hostname file already generated")

            # take max hostnames from hostname file
            all_hostnames = f.readlines()

            logger.info(
                f"Resolving:: {len(all_hostnames)} hostnames on {len(subnets)} subnets"
            )

            # split file
            for index in range(0, len(all_hostnames), chunk_size):
                hostnames = all_hostnames[index : index + chunk_size]

                tmp_hostname_file = create_tmp_csv_file(hostnames)

                logger.info(
                    f"Starting to resolve hostnames {index * chunk_size} to {(index + 1) * chunk_size} (total={len(all_hostnames)})"
                )

                zdns = ZDNS(
                    subnets=subnets,
                    hostname_file=tmp_hostname_file,
                    name_servers="8.8.8.8",
                    table_name=output_table,
                )
                await zdns.main()

                tmp_hostname_file.unlink()
                time.sleep(0.2)

    await raw_dns_mapping(subnets)

    if repeat:
        # TODO: replace with Crontab module
        schedule.every(4).hours.until(end_date).do(raw_dns_mapping)

        while True:
            schedule.run_pending()


def load_json_iter(file_path: Path) -> Generator:
    """iter load json file"""
    with file_path.open("r") as f:
        for row in f.readlines():
            yield json.loads(row)


async def main() -> None:
    """init main"""
    output_table = "dns_mapping_routers_2ms"
    hostname_file = path_settings.DATASET / "hostname_1M_max_bgp_prefix_per_cdn.csv"
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

        if len(subnets) > 10_000:
            break

    subnets = list(subnets)
    await resolve_hostnames(
        subnets=subnets,
        hostname_file=hostname_file,
        output_table=output_table,
        repeat=False,
        end_date=None,
        chunk_size=100,
    )


if __name__ == "__main__":
    asyncio.run(main())
