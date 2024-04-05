import json
import pyasn
import asyncio
import time

from uuid import uuid4
from tqdm import tqdm
from datetime import datetime
from dateutil import parser
from enum import Enum
from pathlib import Path
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import InsertFromCSV, CreateDNSMappingTable

from geogiant.common.files_utils import dump_csv, create_tmp_csv_file
from geogiant.common.ip_addresses_utils import (
    is_valid_ipv4,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from geogiant.common.settings import ZDNSSettings


class ZDNS_STATUS(Enum):
    ERROR = "ERROR"
    NOERROR = "NOERROR"


class ZDNS:
    """ZDNS module for python"""

    def __init__(
        self,
        subnets: list[str],
        hostname_file: Path,
        name_servers: list,
        output_file: Path = None,
        output_table: str = None,
        timeout: float = 0.1,
        iterative: bool = False,
    ) -> None:
        self.subnets = subnets
        self.hostname_file = hostname_file
        self.name_servers = name_servers
        self.output_file = output_file
        self.output_table = output_table
        self.timeout = timeout
        self.iterative = iterative

        self.settings = ZDNSSettings()

    def get_zdns_cmd(self, subnet: str) -> str:
        """parse zdns cmd for a given subnet"""
        hostname_cmd = f"cat {self.hostname_file}"

        if self.iterative:
            return (
                hostname_cmd
                + " | "
                + f"{self.settings.EXEC_PATH} A --client-subnet {subnet} --iterative"
            )
        else:
            return (
                hostname_cmd
                + " | "
                + f"{self.settings.EXEC_PATH} A --client-subnet {subnet} --name-servers {self.name_servers}"
            )

    async def query(self, subnet: str) -> dict:
        """run zdns tool and return zdns raw results"""
        query_results = []

        zdns_cmd = self.get_zdns_cmd(subnet)

        ps = await asyncio.subprocess.create_subprocess_shell(
            zdns_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await ps.communicate()

        if stderr:
            raise RuntimeError(stderr)

        try:
            output = stdout.decode().split("\n")
            for row in output:
                query_result = json.loads(row)
                query_results.append(query_result)

        except json.decoder.JSONDecodeError:
            pass

        return query_results

    def parse(self, subnet: str, query_results: list, asndb) -> list:
        """return resolution server ip addr"""
        parsed_data = []
        for result in query_results:
            if not result["status"] == ZDNS_STATUS.NOERROR.value:
                continue

            # filter result query where some data are missing
            try:
                data = result["data"]
                hostname = result["name"]
                answers = data["answers"]

                timestamp = result["timestamp"]
                timestamp = parser.isoparse(timestamp)
                timestamp = datetime.timestamp(timestamp)

                source_scope = data["additionals"][0]["csubnet"]["source_scope"]
                if source_scope == 0:
                    continue

            except KeyError:
                continue

            # filter answers that are not IP addresses
            for answer in answers:
                answer = answer["answer"]
                if is_valid_ipv4(answer):
                    subnet_addr = get_prefix_from_ip(subnet)
                    answer_subnet = get_prefix_from_ip(answer)
                    answer_asn, answer_bgp_prefix = route_view_bgp_prefix(answer, asndb)

                    if not answer_asn or not answer_bgp_prefix:
                        logger.info(f"{answer}:: Could not retrieve ASN and BGP prefix")
                        continue

                    parsed_data.append(
                        f"{int(timestamp)},\
                        {subnet_addr},\
                        24,\
                        {hostname},\
                        {answer},\
                        {answer_subnet},\
                        {answer_bgp_prefix},\
                        {answer_asn},\
                        {source_scope}"
                    )

        return parsed_data

    async def run(self) -> list:
        """run zdns on a set of client subnet and output data within Clickhouse table"""
        asndb = pyasn.pyasn(str(self.settings.RIB_TABLE))

        zdns_data = []
        for subnet in tqdm(self.subnets):
            query_results = await self.query(subnet)
            parsed_data = self.parse(subnet, query_results, asndb)
            zdns_data.extend(parsed_data)

            if not self.iterative:
                time.sleep(self.timeout)

        return zdns_data

    async def main(self) -> None:
        """ZDNS measurement entrypoint"""

        logger.info(f"ZDNS::Starting resolution on {len(self.subnets)} subnets")

        zdns_data = await self.run()

        tmp_file_path = create_tmp_csv_file(zdns_data)

        if self.output_table:

            async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
                await CreateDNSMappingTable().aio_execute(
                    client=client, table_name=self.output_table
                )

                await InsertFromCSV().execute(
                    table_name=self.output_table,
                    in_file=tmp_file_path,
                )

        if self.output_file:
            output_file = (
                self.output_file.name.split(".")[0] + "_" + str(uuid4()) + ".csv"
            )
            dump_csv(
                data=[data.replace(" ", "") for data in zdns_data],
                output_file=self.settings.RESULTS_PATH / output_file,
            )

        tmp_file_path.unlink()

        logger.info(f"ZDNS::Resolution done")
