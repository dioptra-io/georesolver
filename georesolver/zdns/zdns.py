import json
import pyasn
import asyncio

from uuid import uuid4
from tqdm import trange
from datetime import datetime
from dateutil import parser
from enum import Enum
from pathlib import Path
from loguru import logger
from pprint import pprint
from pych_client import AsyncClickHouseClient

from georesolver.clickhouse import (
    Query,
    CreateDNSMappingTable,
    CreateIPv6DNSMappingTable,
    CreateNameServerTable,
)

from georesolver.common.files_utils import dump_csv, create_tmp_csv_file
from georesolver.common.ip_addresses_utils import (
    is_valid_ipv4,
    is_valid_ipv6,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.settings import ZDNSSettings, PathSettings

path_settings = PathSettings()


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
        request_type: str = "A",
        output_logs: Path = None,
    ) -> None:
        self.subnets = subnets
        self.hostname_file = hostname_file
        self.name_servers = name_servers
        self.output_file = output_file
        self.output_table = output_table
        self.timeout = timeout
        self.iterative = iterative
        self.request_type = request_type
        self.output_logs = output_logs

        self.settings = ZDNSSettings()

    def get_zdns_cmd(self, subnet: str, hostname=None) -> str:
        """parse zdns cmd for a given subnet"""
        hostname_cmd = f"cat {self.hostname_file}"

        if self.iterative:
            if not hostname:
                return (
                    hostname_cmd
                    + " | "
                    + f"{self.settings.EXEC_PATH} {self.request_type} --client-subnet {subnet} --iterative"
                    + " --threads 200"
                )
            else:
                return (
                    f"echo {hostname}"
                    + " | "
                    + f"{self.settings.EXEC_PATH} {self.request_type} --client-subnet {subnet} --iterative"
                    + " --threads 200 --timeout 3"
                )
        else:
            return (
                hostname_cmd
                + " | "
                + f"{self.settings.EXEC_PATH} {self.request_type} --client-subnet {subnet} --name-servers {self.name_servers}"
                + " --threads 200 --timeout 3"
            )

    async def query(self, subnet: str, hostname: str = None) -> dict:
        """run zdns tool and return raw results"""
        query_results = []

        zdns_cmd = self.get_zdns_cmd(subnet, hostname)

        ps = await asyncio.subprocess.create_subprocess_shell(
            zdns_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await ps.communicate()

        if stderr:
            # raise RuntimeError(stderr)
            pass

        try:
            output = stdout.decode().split("\n")
            for row in output:
                query_result = json.loads(row)
                query_results.append(query_result)

        except json.decoder.JSONDecodeError:
            pass

        return query_results, subnet

    def parse_timestamp(self, timestamp: dict) -> datetime:
        """retrieve timestamp from DNS resp"""
        timestamp = parser.isoparse(timestamp)
        timestamp = datetime.timestamp(timestamp)

        return timestamp

    async def parse_a_records(
        self, resp: dict, subnet: str, asndb, hostname: str = None
    ) -> str:
        """parse A records from ZDNS output"""
        parsed_output = []

        try:
            if not hostname:
                hostname = resp["name"]

            resp = resp["results"][self.request_type]

            # check status
            if resp["status"] != "NOERROR":
                return None

            try:
                data = resp["data"]
                answers = data["answers"]
                timestamp = self.parse_timestamp(resp["timestamp"])
                additionals = data["additionals"]

                source_scope = 0
                for additional in additionals:
                    if "csubnet" not in additional:
                        continue

                    source_scope = additional["csubnet"]["source_scope"]
                    subnet = additional["csubnet"]["address"]
                    break

                if source_scope == 0:
                    return None

            except Exception as e:
                return None

        except KeyError:
            return None

        for answer in answers:
            answer = answer["answer"]
            if is_valid_ipv4(answer) and self.request_type == "A":
                subnet_addr = get_prefix_from_ip(subnet)
                answer_subnet = get_prefix_from_ip(answer)
            elif is_valid_ipv6(answer) and self.request_type == "AAAA":
                subnet_addr = get_prefix_from_ip(subnet, ipv6=True)
                answer_subnet = get_prefix_from_ip(answer, ipv6=True)
            else:
                continue

            answer_asn, answer_bgp_prefix = route_view_bgp_prefix(answer, asndb)

            if not answer_asn or not answer_bgp_prefix:
                answer_asn = -1
                answer_bgp_prefix = "None"

            string_data = f"{int(timestamp)},\
                {subnet_addr},\
                {24 if self.request_type == 'A' else 56},\
                {hostname},\
                {answer},\
                {answer_subnet},\
                {answer_bgp_prefix},\
                {answer_asn},\
                {source_scope}"

            parsed_output.append(string_data)

        return parsed_output

    def parse_ns_records(self, resp: dict) -> str:
        """parse NS records from ZDNS output"""
        parsed_output = []

        try:
            hostname = resp["name"]
            resp = resp["results"][self.request_type]

            if resp["status"] != "NOERROR":
                return None

            data = resp["data"]
            timestamp = self.parse_timestamp(resp["timestamp"])

        except KeyError as e:
            return None

        if "answers" in data:
            answers = data["answers"]
            for answer in answers:
                if answer["type"] != "NS":
                    continue

                ns = answer["answer"]

                if not ns:
                    continue

                parsed_output.append(
                    f"{int(timestamp)},\
                    {hostname},\
                    {ns},\
                    {answer['type']}"
                )

        if "authorities" in data:
            authorities = data["authorities"]
            for auth in authorities:
                ns = auth["ns"]

                if not ns:
                    continue

                parsed_output.append(
                    f"{int(timestamp)},\
                    {hostname},\
                    {ns},\
                    {auth['type']}"
                )

        return parsed_output

    async def parse(self, subnet: str, query_results: list, asndb) -> list:
        """return resolution server ip addr"""
        parsed_data = []
        for resp in query_results:

            # filter answers that are not IP addresses
            if self.request_type == "A" or self.request_type == "AAAA":
                data = await self.parse_a_records(resp, subnet, asndb)

            elif self.request_type == "NS":
                data = self.parse_ns_records(resp)

            else:
                raise RuntimeError(f"DNS Type:: {self.request_type} is unkown")

            if data:
                parsed_data.extend(data)

        return parsed_data

    async def run(self) -> list:
        """run zdns on a set of client subnet and output data within Clickhouse table"""
        asndb = pyasn.pyasn(str(self.settings.RIB_TABLE))

        zdns_data = []

        if self.output_logs:
            output_file = open(self.output_logs, "a")
        else:
            output_file = None

        step_size = 3
        for i in trange(0, len(self.subnets), step_size, file=output_file):
            batch_subnets = self.subnets[i : i + step_size]
            tasks = tuple([self.query(subnet) for subnet in batch_subnets])

            query_results = await asyncio.gather(*tasks)

            for result, subnet in query_results:
                parsed_data = await self.parse(subnet, result, asndb)
                zdns_data.extend(parsed_data)

        if output_file:
            output_file.close()

        return zdns_data

    async def main(self) -> None:
        """ZDNS measurement entrypoint"""

        logger.info(f"ZDNS::Starting resolution on {len(self.subnets)} subnets")

        zdns_data = await self.run()

        logger.info(f"Resolution done, inserting to :: {self.output_table}")

        tmp_file_path = create_tmp_csv_file(zdns_data)

        if self.output_table:
            async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
                if self.request_type == "A":
                    await CreateDNSMappingTable().aio_execute(
                        client=client, table_name=self.output_table
                    )
                if self.request_type == "AAAA":
                    await CreateIPv6DNSMappingTable().aio_execute(
                        client=client, table_name=self.output_table
                    )
                elif self.request_type == "NS":
                    await CreateNameServerTable().aio_execute(
                        client=client, table_name=self.output_table
                    )

                Query().execute_insert(
                    client=client,
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


# testting, debugging
if __name__ == "__main__":

    import asyncio
    from georesolver.zdns import ZDNS

    subnets = ["132.227.123.0/24"]
    hostname_file = path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv"

    zdns = ZDNS(
        subnets=subnets,
        hostname_file=hostname_file,
        output_table="test_ecs_mapping",
        name_servers="8.8.8.8",
        request_type="A",
    )

    asyncio.run(zdns.main())
