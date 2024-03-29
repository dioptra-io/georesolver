import asyncio

from typing import Generator
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from loguru import logger
from pych_client import AsyncClickHouseClient, ClickHouseClient
from clickhouse_driver import Client

from geogiant.common.settings import ClickhouseSettings


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
            logger.info(f"{cmd}::Successfully executed")

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

    settings = ClickhouseSettings()

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

        logger.info(f"query={self.name} table_name={table_name}  limit={limit}")
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        rows += await client.json(statement, data=data, settings=settings)

        return rows

    def execute(
        self,
        client: ClickHouseClient,
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

        logger.info(f"query={self.name} table_name={table_name}  limit={limit}")
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

        logger.info(f"query={self.name} table_name={table_name}  limit={limit}")
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

        logger.info(f"query={self.name} table_name={table_name} limit={limit}")
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
        logger.info(f"query={self.name} table_name={table_name} limit={limit}")
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        return client.iter_json(statement, data=data, settings=settings)


@dataclass(frozen=True)
class InsertCSV(Query):
    def statement(self, table_name: str) -> str:
        return f"INSERT INTO {self.settings.DATABASE}.{table_name} FORMAT CSV"


@dataclass(frozen=True)
class Drop(Query):
    def statement(self, table_name: str) -> str:
        return f"DROP TABLE {self.settings.DATABASE}.{table_name}"


# clickhouse files cannot be sent directly from the server
@dataclass(frozen=True)
class InsertFromInFile:
    settings = ClickhouseSettings()

    def statement(self, table_name: str, in_file: Path) -> str:
        return f"INSERT INTO {self.settings.DATABASE}.{table_name} FROM INFILE '{in_file}' FORMAT Native"

    async def execute(self, table_name: str, in_file: Path) -> None:
        """insert data contained in local file"""
        cmd = f'clickhouse client --query="{self.statement(table_name, in_file)}"'

        ps = await asyncio.subprocess.create_subprocess_shell(
            cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        _, stderr = await ps.communicate()

        if stderr:
            raise RuntimeError(
                f"Could not insert data::{cmd}, failed with error: {stderr}"
            )
        else:
            logger.info(f"{cmd}::Successfully executed")


@dataclass(frozen=True)
class InsertFromCSV:
    settings = ClickhouseSettings()

    def statement(self, table_name: str, in_file: Path) -> str:
        return f"INSERT INTO {self.settings.DATABASE}.{table_name} FROM INFILE '{in_file}' FORMAT CSV"

    async def execute(self, table_name: str, in_file: Path) -> None:
        """insert data contained in local file"""
        cmd = f'clickhouse client --query="{self.statement(table_name, in_file)}"'

        ps = await asyncio.subprocess.create_subprocess_shell(
            cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        _, stderr = await ps.communicate()

        if stderr:
            raise RuntimeError(
                f"Could not insert data::{cmd}, failed with error: {stderr}"
            )
        else:
            logger.info(f"{cmd}::Successfully executed")
