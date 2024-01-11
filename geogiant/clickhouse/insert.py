"""clickhouse utilities"""
import subprocess

from uuid import uuid4
from pathlib import Path
from clickhouse_driver import Client
from abc import abstractmethod
from loguru import logger

from geogiant.clickhouse import Clickhouse


class Insert(Clickhouse):
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

    @classmethod
    def insert(
        self,
        input_data: list[str],
        table_name: str,
        create_table_statement: str,
        drop_table: bool = False,
    ) -> None:
        """insert data into clickhouse base method"""
        # parse output table name
        table_name = str(table_name).replace("-", "_")

        # create a temp csv file result
        tmp_file_dir = self.common_settings.TMP_PATH / f"{uuid4()}.csv"
        self.common_settings.TMP_PATH.mkdir(parents=True, exist_ok=True)

        # drop table if asked
        if drop_table:
            logger.warning(f"{table_name}::was dropped")
            statement = self.drop_table_statement(table_name)
            self.execute_iter(statement)

        logger.debug(f"{table_name}::Inserting {len(input_data)} rows")

        # create table if not exists
        self.create_table(create_table_statement)

        # write tmp csv file
        self.write_tmp(input_data, tmp_file_dir)

        # insert results into db from file
        statement = self.insert_from_csv_statement(
            table_name=table_name,
            input_file_dir=tmp_file_dir,
        )
        self.execute_insert(statement)

        # remove tmp file
        tmp_file_dir.unlink()
