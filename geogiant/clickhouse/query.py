from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.common.settings import ClickhouseSettings


@dataclass(frozen=True)
class Query:
    """Base class for every query."""

    settings = ClickhouseSettings()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def statement(self, table_name: str) -> str:
        # As a query user, prefer calling `statements` instead of `statement` as there
        # is no guarantees that the query will implement this method and return a single statement.
        raise NotImplementedError

    async def execute(
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
        logger.info(f"Executing::{statement}")
        rows += await client.json(statement, data=data, settings=settings)

        return rows

    def execute_iter(
        self,
        client: AsyncClickHouseClient,
        table_name: str,
        data: Any = None,
        limit=None,
    ) -> Iterator[dict]:
        """
        Execute the query and return each row as a dict, as they are received from the database.
        """
        statement = self.statement(table_name)

        logger.info(f"query={self.name} table_name={table_name} limit={limit}")
        settings = dict(
            limit=limit[0] if limit else 0,
            offset=limit[1] if limit else 0,
        )
        yield from client.iter_json(statement, data=data, settings=settings)


class Insert(Query):
    def statement(self, table_name: str) -> str:
        return f"INSERT INTO {self.settings.DB}.{table_name} FORMAT CSV"
