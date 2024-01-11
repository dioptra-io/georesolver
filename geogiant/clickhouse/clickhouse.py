"""clickhouse utilities"""
from clickhouse_driver import Client
from abc import ABC
from loguru import logger

from geogiant.common.settings import PathSettings, ClickhouseSettings


class Clickhouse(ABC):
    """abstract class for clickhouse db interaction"""

    settings = ClickhouseSettings()
    common_settings = PathSettings()

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
