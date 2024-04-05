import json
import pyasn
import asyncio

from tqdm import tqdm
from datetime import datetime
from dateutil import parser
from enum import Enum
from pathlib import Path
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import InsertFromCSV, Query

from geogiant.common.files_utils import create_tmp_csv_file, iter_file
from geogiant.common.ip_addresses_utils import (
    is_valid_ipv4,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


class FingerprintDNS(Query):
    def statement(self, table_name: str) -> str:
        return f"""
        WITH length(arrayIntersect(t1.mapping, t2.mapping)) / least(length(t1.mapping), length(t2.mapping)) as fingerprint
        SELECT
            (client_subnet, groupUniqArray(answer_bgp_prefix)) as vps_mapping
        FROM
            {self.settings.DATABASE}.{table_name}
        GROUP BY
            client_subnet
        """


def test() -> None:
    for i in range(10):
        yield i


async def async_test() -> None:
    async for i in range(10):
        yield i


async def test_main() -> None:
    async for j in async_test():
        yield print(j)


async def main() -> None:
    tmp_file = path_settings.TMP_PATH / "tmp__aa231144-cc69-48d7-90af-f3e053e44eaa.csv"

    # output results
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        # await CreateDNSMappingTable().execute(
        #     client, clickhouse_settings.DNS_MAPPING_VPS_RAW
        # )

        await InsertFromCSV().execute(
            table_name=clickhouse_settings.DNS_MAPPING_VPS_RAW,
            in_file=tmp_file,
        )


# if __name__ == "__main__":
#     asyncio.run(main())
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

from ipwhois import IPWhois


def get_cdn_info(as_number):

    ipwhois_result = IPWhois(as_number)
    results = ipwhois_result.lookup_rdap()
    logger.info(results.keys())

    for key, val in results.items():
        logger.info(f"{key} = {val}")

        if key == "objects":
            for key, val in results["objects"].items():
                print(key, val)

    for entity in entities:
        logger.debug(entity)
        roles = entity.get("roles", [])
        if "content" in roles:
            return entity["handle"]

    return "CDN information not found for AS{}".format(as_number)


if __name__ == "__main__":
    as_number = "142.250.70.142"
    # as_number = "162.245.85.194"
    # as_number = "23.51.84.160"
    # as_number = "23.205.185.245"
    as_number = "31.13.87.1"

    cdn_info = get_cdn_info(as_number)
    print(cdn_info)
