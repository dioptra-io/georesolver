import json
import subprocess

from datetime import datetime
from dateutil import parser
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from loguru import logger

from common.ip_addresses_utils import is_ipv4, get_prefix_from_ip
from insert_results import InsertMappingDNS
from settings import ZDNSSettings

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
        output_table: str,
    ) -> None:
        self.subnets = subnets
        self.hostname_file = hostname_file
        self.name_servers = name_servers
        self.output_table = output_table
        
        self.settings = ZDNSSettings()
                
    def get_hostname_cmd(self) -> list:
        """return the command to ouput procress file into zdns"""
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
    
    def parse_data(self, subnet: str, query_results: list) -> list:
        """return resolution server ip addr"""
        parsed_data = []
        for result in query_results:
            if not result["status"] == ZDNS_STATUS.NOERROR.value:
                continue
            
            # filter result query where some data are missing
            try:
                data = result["data"]
                hostname = result["name"]
                
                timestamp = result["timestamp"]    
                timestamp = parser.isoparse(timestamp)
                timestamp = datetime.timestamp(timestamp)       
                    
                answers = data["answers"]
                
                # if multiple source scope are found, consider invalid answer
                source_scope = [metadata["csubnet"]["source_scope"] for metadata in  data["additionals"]]
                if len(source_scope) > 1:
                    continue
                else:
                    source_scope=source_scope[0]
                    
        
            except KeyError as e:
                logger.error(f"Invalid zdns query results for IP addr: {subnet}, {e}")
                continue
            
            # filter answers that are not IP addresses
            for answer in answers:
                answer = answer['answer']
                if is_ipv4(answer):
                    subnet_addr = get_prefix_from_ip(subnet)
                    parsed_data.append(
                        f"{int(timestamp)},{subnet_addr},24,{hostname},{answer}"
                    )
        return parsed_data

    def run(
        self,
    ) -> None:
        """run zdns on a set of client subnet and output data within Clickhouse table
        """        
        # run ZDNS on input subnets
        zdns_data = []
        for subnet in self.subnets:
            query_results = self.query(subnet)
            parsed_data = self.parse_data(subnet, query_results)
            zdns_data.extend(parsed_data)
                        
        # output results
        InsertMappingDNS().insert(
            input_data=zdns_data,
            output_table=self.output_table,
        )
