import asyncio
import subprocess

from typing import Generator
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from loguru import logger
from pych_client import AsyncClickHouseClient, ClickHouseClient
from clickhouse_driver import Client

from georesolver.common.settings import ClickhouseSettings


@dataclass(frozen=True)
class NativeQuery:
    settings = ClickhouseSettings()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def statement(self, table_name: str, **kwargs) -> str:
        raise NotImplementedError

    async def execute(self, table_name: str, **kwargs) -> None:
        """insert data contained in local file"""
        cmd = f'clickhouse client --query="{self.statement(table_name, **kwargs)}"'

        ps = await asyncio.subprocess.create_subprocess_shell(
            cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        stdout, stderr = await ps.communicate()

        if stderr:
            raise RuntimeError(
                f"Could not insert data::{cmd}, failed with error: {stderr}"
            )
        else:
            logger.debug(f"{cmd}::Successfully executed")

        stdout = stdout.decode()

        data = []
        for row in stdout.splitlines():
            row = row.split("\t")
            data.append(row)

        return data

    def execute_iter(self, table_name: str, **kwargs) -> Generator:
        """execute select statement with clickhouse driver"""
        statement = self.statement(table_name, **kwargs)

        client = Client(host="localhost")
        yield from client.execute_iter(statement)


@dataclass(frozen=True)
class Query:
    """Base class for every query."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def statement(self, table_name: str, **kwargs) -> str:
        raise NotImplementedError

    async def aio_execute(
        self,
        client: AsyncClickHouseClient,
        table_name: str,
        data: Any = None,
        limit=None,
        **kwargs,
    ) -> list[dict]:
        """
        Execute the query and return each row as a dict.
        Args:
            client: ClickHouse client.
            measurement_id: Measurement id.
            data: str or bytes iterator containing data to send.
            limit: (limit, offset) tuple.
            subsets: Iterable of IP networks on which to execute the query independently.
        """
        rows = []
        statement = self.statement(table_name, **kwargs)

        logger.debug(f"query={self.name} table_name={table_name}  limit={limit}")
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        rows += await client.json(statement, data=data, settings=settings)

        return rows

    def execute(
        self,
        client: ClickHouseClient,
        table_name: str = None,
        data: Any = None,
        limit=None,
        **kwargs,
    ) -> list[dict]:
        """
        Execute the query and return each row as a dict.
        Args:
            client: ClickHouse client.
            measurement_id: Measurement id.
            data: str or bytes iterator containing data to send.
            limit: (limit, offset) tuple.
            subsets: Iterable of IP networks on which to execute the query independently.
        """
        rows = []
        statement = self.statement(table_name=table_name, **kwargs)
        database = (
            client.config["database"]
            if not "database" in kwargs
            else kwargs["database"]
        )

        logger.debug(
            f"query={self.name}; database={database}; table_name={table_name}  limit={limit}"
        )

        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        rows += client.json(statement, data=data, settings=settings)

        return rows

    async def execute_bytes(
        self,
        client: AsyncClickHouseClient,
        table_name: str,
        data: Any = None,
        limit=None,
    ) -> list[dict]:
        """
        Execute the query and return each row as a dict.
        Args:
            client: ClickHouse client.
            measurement_id: Measurement id.
            data: str or bytes iterator containing data to send.
            limit: (limit, offset) tuple.
            subsets: Iterable of IP networks on which to execute the query independently.
        """
        rows = []
        statement = self.statement(table_name)

        logger.debug(f"query={self.name} table_name={table_name}  limit={limit}")
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        rows += await client.bytes(statement, data=data, settings=settings)

        return rows

    async def aio_execute_iter(
        self,
        client: AsyncClickHouseClient,
        table_name: str,
        data: Any = None,
        limit=None,
        **kwargs,
    ) -> Iterator[dict]:
        """
        Execute the query and return each row as a dict, as they are received from the database.
        """
        statement = self.statement(table_name, **kwargs)

        logger.debug(f"query={self.name} table_name={table_name} limit={limit}")
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        return client.iter_json(statement, data=data, settings=settings)

    def execute_iter(
        self,
        client: ClickHouseClient,
        table_name: str,
        data: Any = None,
        limit=None,
        **kwargs,
    ) -> Iterator[dict]:
        """
        Execute the query and return each row as a dict, as they are received from the database.
        """
        statement = self.statement(table_name, **kwargs)
        logger.debug(
            f"query={self.name}; database={client.config['database']} table_name={table_name} limit={limit}"
        )
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        return client.iter_json(statement, data=data, settings=settings)

    def execute_insert(
        self,
        client: ClickHouseClient,
        table_name: str,
        in_file: Path,
        format: str = "CSV",
        out_db: str = ClickhouseSettings().CLICKHOUSE_DATABASE,
    ):
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            query = f"""
            INSERT INTO 
                {out_db}.{table_name} 
            FORMAT 
                {format}
            """
            client.execute(query, data=in_file.read_bytes())

    def execute_extract(
        self, client: ClickHouseClient, table_name: str, out_file: Path
    ):
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            query = f"""
            SELECT 
                * 
            FROM 
                {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name} 
            INTO OUTFILE 
                '{out_file}'
            FORMAT 
                Native
            """
            client.execute(query)


@dataclass(frozen=True)
class InsertCSV(Query):
    def statement(self, table_name: str) -> str:
        return f"INSERT INTO {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name} FORMAT CSV"


@dataclass(frozen=True)
class DropTable(Query):
    def statement(self, table_name: str) -> str:
        return f"DROP TABLE {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}"


# clickhouse files cannot be sent directly from the server
@dataclass(frozen=True)
class InsertFromInFile:
    settings = ClickhouseSettings()

    def statement(
        self,
        table_name: str,
        in_file: Path,
        database: str = ClickhouseSettings().CLICKHOUSE_DATABASE,
    ) -> str:
        return (
            f"INSERT INTO {database}.{table_name} FROM INFILE '{in_file}' FORMAT Native"
        )

    def execute(
        self,
        table_name: str,
        in_file: Path,
        database: str = ClickhouseSettings().CLICKHOUSE_DATABASE,
    ) -> None:
        """insert data contained in local csv file"""
        cmd = f"""clickhouse client \
            --user {ClickhouseSettings().CLICKHOUSE_USERNAME} \ 
            --password {ClickhouseSettings().CLICKHOUSE_PASSWORD} \
            --protocol http \
            --query=\"{self.statement(table_name, in_file,database)}\"
        """

        ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        logger.debug(f"Insert from CSV:: {in_file=}, {ps.stdout=}, {ps.stderr=}")

        if ps.stderr:
            raise RuntimeError(f"Could not insert data::{cmd}, error:: {ps.stderr}")
        else:
            logger.info(f"{cmd}::Successfully executed")


@dataclass(frozen=True)
class ExtractTableData:
    def statement(self, table_name: str, out_file: Path) -> str:
        return f"""
        SELECT
            *
        FROM
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        INTO OUTFILE
            '{out_file}'
        FORMAT
            Native
        """

    def execute(self, table_name: str, out_file: Path) -> None:
        """execute query with native client"""
        cmd = f"clickhouse client"
        cmd += f" --user {ClickhouseSettings().CLICKHOUSE_USERNAME}"
        cmd += f" --password {ClickhouseSettings().CLICKHOUSE_PASSWORD}"
        cmd += f' --query="{self.statement(table_name, out_file)}"'

        ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        logger.debug(
            f"Extract data from {table_name=} to {out_file=} {ps.stdout=}, {ps.stderr=}"
        )

        if ps.stderr:
            raise RuntimeError(f"Could execute query, error:: {ps.stderr}")
        else:
            logger.info(f"::Successfully executed")


@dataclass(frozen=True)
class InsertFromCSV:
    settings = ClickhouseSettings()

    def statement_deprec(self, table_name: str, in_file: Path) -> str:
        return f"INSERT INTO {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name} FROM INFILE '{in_file}' FORMAT CSV"

    def statement(self, table_name: str, in_file: Path) -> str:
        return f"""
        INSERT INTO FUNCTION 
            file('{in_file}', 'Native', 'ZSTD')
        SELEC
            *
        FROM 
            {ClickhouseSettings().CLICKHOUSE_DATABASE}.{table_name}
        """

    def execute(self, table_name: str, in_file: Path) -> None:
        """insert data contained in local csv file"""
        cmd = f"""clickhouse client \
            --host {ClickhouseSettings().CLICKHOUSE_HOST} \ 
            --port {ClickhouseSettings().CLICKHOUSE_PORT} \
            --protocol http \
            --query=\"{self.statement(table_name, in_file)}\"
        """

        ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        logger.debug(f"Insert from CSV:: {in_file=}, {ps.stdout=}, {ps.stderr=}")

        if ps.stderr:
            raise RuntimeError(f"Could not insert data::{cmd}, error:: {ps.stderr}")
        else:
            logger.info(f"{cmd}::Successfully executed")

    async def aio_execute(self, table_name: str, in_file: Path) -> None:
        """insert data contained in local file asynchronously"""
        cmd = f'clickhouse client --query="{self.statement(table_name, in_file)}"'

        ps = await asyncio.subprocess.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await ps.communicate()

        if stderr:
            logger.error(stderr)
            stderr = stderr.decode().splitlines()
            for row in stderr:
                logger.error(row.strip())
            raise RuntimeError(f"Could not insert data::{cmd}")
        else:
            logger.info(f"{cmd}::Successfully executed")
