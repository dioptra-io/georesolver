"""clickhouse utilities"""
import subprocess

from uuid import uuid4
from pathlib import Path
from clickhouse_driver import Client
from abc import ABC, abstractmethod
from loguru import logger

from common.settings import CommonSettings, ClickhouseSettings

class Clickhouse(ABC):
    """abstract class for clickhouse db interaction"""
    
    settings = ClickhouseSettings()
    common_settings = CommonSettings()
    
    @classmethod
    def to_ipv6(self, ipv4_addr: str) -> str:
        """add ipv6 format to ipv4 for unified storage"""
        return "::ffff:" + ipv4_addr

    @classmethod
    def to_ipv4(self, ipv6_addr: str) -> str:
        """extract IPv6 embedded IPv4 addr"""
        return ipv6_addr.split("::ffff:")[-1]

    @classmethod
    def drop_table_statement(self, table_name: str) -> None:
        """drop table from table name"""
        return f"DROP TABLE {self.settings.CLICKHOUSE_DB}.{table_name}"

    @classmethod
    def execute(self, statement: str) -> list:
        """execute select statement with clickhouse driver"""

        client = Client(host=self.settings.CLICKHOUSE_HOST)
        req_results = client.execute(statement)

        return req_results

    @classmethod
    def execute_iter(self, statement: str) -> list:
        """execute select statement with clickhouse driver"""
        logger.debug(f"executing query: {statement}")

        client = Client(host=self.settings.CLICKHOUSE_HOST)
        req_results = client.execute_iter(statement)

        return req_results

    @classmethod
    def execute_dataframe(self, statement: str) -> dict:
        """execute select statement with clickhouse driver"""
        logger.debug(f"executing query: {statement}")

        client = Client(host=self.settings.CLICKHOUSE_HOST)
        req_results = client.query_dataframe(statement)

        return req_results


class ClickhouseInsert(Clickhouse):
    """abstract class for clickhouse db interaction"""
    @abstractmethod
    def create_table_statement(self, table_name: str) -> str:
        """returns anchors mapping table query"""
        raise NotImplementedError("Create table statement not implemented")
    
    @classmethod
    def create_table(self, statement: str) -> None:
        """create anchor mapping table"""
        logger.debug(f"executing query: {statement}")
        
        client = Client(host=self.settings.CLICKHOUSE_HOST)
        client.execute(statement)

    @classmethod
    def insert_from_csv_statement(self, table_name: str, input_file_dir: Path) -> None:
        """returns insert query for anchor mapping results"""
        return f"""
        INSERT INTO {self.settings.CLICKHOUSE_DB}.{table_name}
        FROM INFILE \'{str(input_file_dir)}\'
        FORMAT CSV
        """
        
    @classmethod
    def execute_insert(self, statement: str):
        """execute insert statement using subprocesses
        (clickhouse driver does not support well insert)
        """
        cmd = [
            "clickhouse-client",
            f"--host={self.settings.CLICKHOUSE_HOST}",
            f"--query={statement}",
        ]

        logger.debug(f"executing query: {cmd}")

        # execute clickhouse process with query, TODO: add settings
        process = subprocess.Popen(cmd)
        process.wait()
        
    @classmethod
    def write_tmp(self, data: list[str], output_file: Path) -> None:
        """
        write temporary csv file results with name for clickhouse insertion

        """
        with output_file.open("w") as f:
            for row_data in data:
                f.write(f"{row_data}\n")