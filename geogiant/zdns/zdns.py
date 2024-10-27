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
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import (
    Query,
    CreateDNSMappingTable,
    CreateNameServerTable,
)

from geogiant.common.files_utils import dump_csv, create_tmp_csv_file
from geogiant.common.ip_addresses_utils import (
    is_valid_ipv4,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from geogiant.common.settings import ZDNSSettings, PathSettings

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

    def get_zdns_cmd(self, subnet: str) -> str:
        """parse zdns cmd for a given subnet"""
        hostname_cmd = f"cat {self.hostname_file}"

        if self.iterative:
            return (
                hostname_cmd
                + " | "
                + f"{self.settings.EXEC_PATH} {self.request_type} --client-subnet {subnet} --iterative"
            )
        else:
            return (
                hostname_cmd
                + " | "
                + f"{self.settings.EXEC_PATH} {self.request_type} --client-subnet {subnet} --name-servers {self.name_servers}"
                + " --threads 200 --timeout 3"
            )

    async def query(self, subnet: str) -> dict:
        """run zdns tool and return raw results"""
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

        return query_results, subnet

    def parse_timestamp(self, resp: dict) -> datetime:
        """retrieve timestamp from DNS resp"""
        timestamp = resp["timestamp"]
        timestamp = parser.isoparse(timestamp)
        timestamp = datetime.timestamp(timestamp)

        return timestamp

    def parse_a_records(self, resp: dict, subnet: str, asndb) -> str:
        """parse A records from ZDNS output"""
        parsed_output = []
        try:
            hostname = resp["name"]
            results = resp["results"]["A"]

            if not results["status"] == ZDNS_STATUS.NOERROR.value:
                return None

            answers = answers = results["data"]["answers"]
            timestamp = self.parse_timestamp(results)
            source_scope = results["data"]["additionals"][0]["csubnet"]["source_scope"]

            if source_scope == 0:
                return None

        except KeyError:
            return None

        for answer in answers:

            # check answer type
            if answer["type"] != "A":
                continue

            answer = answer["answer"]
            if is_valid_ipv4(answer):
                subnet_addr = get_prefix_from_ip(subnet)
                answer_subnet = get_prefix_from_ip(answer)
                answer_asn, answer_bgp_prefix = route_view_bgp_prefix(answer, asndb)

                if not answer_asn or not answer_bgp_prefix:
                    logger.debug(f"{answer}:: Could not retrieve ASN and BGP prefix")
                    answer_asn = -1
                    answer_bgp_prefix = "None"

                parsed_output.append(
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

        return parsed_output

    def parse_ns_records(
        self,
        resp: dict,
        subnet: str,
    ) -> str:
        """parse NS records from ZDNS output"""
        parsed_output = []
        try:
            resp_body = resp["data"]
            timestamp = self.parse_timestamp(resp)
            hostname = resp["name"]

        except KeyError as e:
            return None

        if "answers" in resp_body:
            answers = resp_body["answers"]
            for answer in answers:
                if answer["type"] != "NS":
                    continue

                ns = answer["answer"]

                if not ns:
                    continue

                subnet_addr = get_prefix_from_ip(subnet)

                parsed_output.append(
                    f"{int(timestamp)},\
                    {subnet_addr},\
                    24,\
                    {hostname},\
                    {ns},\
                    {answer['type']}"
                )

        if "authorities" in resp_body:
            authorities = resp_body["authorities"]
            for auth in authorities:
                ns = auth["ns"]

                if not ns:
                    continue

                subnet_addr = get_prefix_from_ip(subnet)

                parsed_output.append(
                    f"{int(timestamp)},\
                    {subnet_addr},\
                    24,\
                    {hostname},\
                    {ns},\
                    {auth['type']}"
                )

        return parsed_output

    def parse(self, subnet: str, query_results: list, asndb) -> list:
        """return resolution server ip addr"""
        parsed_data = []
        for resp in query_results:

            # filter answers that are not IP addresses
            if self.request_type == "A":
                data = self.parse_a_records(resp, subnet, asndb)

            elif self.request_type == "NS":
                data = self.parse_ns_records(resp, subnet)

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

        step_size = 2
        for i in trange(0, len(self.subnets), step_size, file=output_file):
            batch_subnets = self.subnets[i : i + step_size]
            tasks = tuple([self.query(subnet) for subnet in batch_subnets])

            query_results = await asyncio.gather(*tasks)

            for result, subnet in query_results:
                parsed_data = self.parse(subnet, result, asndb)
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
    from geogiant.zdns import ZDNS

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
