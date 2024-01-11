import json
import subprocess

from tqdm import tqdm
from pyasn import pyasn
from datetime import datetime
from dateutil import parser
from enum import Enum
from pathlib import Path
from loguru import logger

from clickhouse import DNSMappingTable

from common.ip_addresses_utils import (
    is_valid_ipv4,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from common.settings import ZDNSSettings


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
        table_name: str,
    ) -> None:
        self.subnets = subnets
        self.hostname_file = hostname_file
        self.name_servers = name_servers
        self.table_name = table_name

        self.settings = ZDNSSettings()

    def get_hostname_cmd(self) -> list:
        """return the command to ouput process file into zdns"""
        return ["cat", f"{self.hostname_file}"]

    def get_zdns_cmd(self, subnet: str) -> list:
        """parse and return zdns cmd"""
        return [
            f"{self.settings.EXEC_PATH}",
            f"A",
            "--client-subnet",
            f"{subnet}",
            "--name-servers",
            f"{self.name_servers}",
        ]

    def query(self, subnet: str) -> dict:
        """run zdns tool and return zdns raw results"""
        query_results = []

        hostname_cmd = self.get_hostname_cmd()
        zdns_cmd = self.get_zdns_cmd(subnet)

        # run zdns with pipe, TODO: Not bad, not great, good enough...
        ps = subprocess.Popen(hostname_cmd, stdout=subprocess.PIPE)
        output = subprocess.check_output(zdns_cmd, stdin=ps.stdout)
        ps.wait()

        # get result
        try:
            output = output.decode().split("\n")
            for row in output:
                query_result = json.loads(row)
                query_results.append(query_result)

        except json.decoder.JSONDecodeError:
            pass

        return query_results

    def parse(self, subnet: str, query_results: list, asndb: pyasn) -> list:
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
                if source_scope == 0 and source_scope == 32:
                    continue

            except KeyError as e:
                logger.error(f"Invalid zdns query results for IP addr: {subnet}, {e}")
                continue

            # filter answers that are not IP addresses
            for answer in answers:
                answer = answer["answer"]
                if is_valid_ipv4(answer):
                    subnet_addr = get_prefix_from_ip(subnet)
                    answer_asn, answer_bgp_prefix = route_view_bgp_prefix(answer, asndb)

                    if not answer_asn or not answer_bgp_prefix:
                        logger.info(f"{answer}::Could not retrieve ASN and BGP prefix")

                    parsed_data.append(
                        f"{int(timestamp)},\
                        {subnet_addr},\
                        24,\
                        {hostname},\
                        {answer},\
                        {answer_asn},\
                        {answer_bgp_prefix}"
                    )

        return parsed_data

    def run(self) -> list:
        """run zdns on a set of client subnet and output data within Clickhouse table"""
        asndb = pyasn(str(self.settings.RIB_TABLE))

        # run ZDNS on input subnets
        zdns_data = []
        for subnet in tqdm(self.subnets):
            query_results = self.query(subnet)
            parsed_data = self.parse(subnet, query_results, asndb)
            zdns_data.extend(parsed_data)

        return zdns_data

    def main(self) -> None:
        """ZNDS measurement entrypoint"""

        logger.info(f"ZDNS::Starting resolution on {len(self.subnets)} subnets")

        zdns_data = self.run()

        create_table_statement = DNSMappingTable().create_table_statement(
            self.table_name
        )

        # output results
        DNSMappingTable().insert(
            input_data=zdns_data,
            table_name=f"{self.table_name}",
            create_table_statement=create_table_statement,
        )

        logger.info(f"ZDNS::Resolution done")
