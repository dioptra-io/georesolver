"""this script is meant to test if a local machine is able to support queries that imply large chunk of data"""

from loguru import logger
from pych_client import ClickHouseClient


from georesolver.clickhouse import GetPingsPerTargetExtended
from georesolver.common.settings import ClickhouseSettings

ch_settings = ClickhouseSettings()


def test_loading() -> None:
    """direct loading without using chunk of data"""
    with ClickHouseClient(**ch_settings.clickhouse) as client:
        rows = GetPingsPerTargetExtended().execute_iter(
            client, table_name="meshed_cdns_pings"
        )

        pings_per_target = []
        for row in rows:
            pings_per_target.append(row)

    logger.info(f"Loaded {len(pings_per_target)} pings per target")


def test_loading_chunk() -> None:
    """test loading on chunk of data to avoid exceeding memory"""
    # TODO: load tables chunk by chunk based on the total number of lines of the table
    # then concatenate results and return
    pass


def main() -> None:
    """simply execute a query that load an important ammount of data in clickhouse"""
    test_loading()

    test_loading_chunk()


if __name__ == "__main__":
    main()
