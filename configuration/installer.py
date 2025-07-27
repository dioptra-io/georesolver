"""extract, insert data into the right tables for evaluation and measurements"""

import typer
import subprocess

from enum import Enum
from pathlib import Path
from loguru import logger
from pych_client import ClickHouseClient

from georesolver.clickhouse import (
    ExtractTableData,
    CreateVPsTable,
    CreatePingTable,
    CreateScoreTable,
    CreateGeolocTable,
    CreateScheduleTable,
    CreateDNSMappingTable,
    CreateNameServerTable,
    CreateTracerouteTable,
)
from georesolver.clickhouse.queries import get_tables
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

DEBUG: bool = True  # set to false if not debugging
if DEBUG:
    OUT_DB: str = "GeoResolver_dev"
else:
    OUT_DB: str = "GeoResolver"


class TableTypes(Enum):
    DNS: str = "dns"
    Geoloc: str = "geoloc"
    Ping: str = "ping"
    NS: str = "name_server"
    Schedule: str = "schedule"
    Score: str = "score"
    VPs: str = "vps"
    Traceroute: str = "traceroute"


def upload_to_ftp(tables: list[dict[str, str]]) -> None:
    """upload_to_ftp data from an input list of tables/output files"""
    for table in tables:
        table_name = table["table_name"]
        out_file = Path(table["output_path"]) / (table_name + ".zst")
        logger.info(f"{table_name=}; {out_file=}")

        if out_file.exists():
            logger.warning(f"File {out_file=} already exists")
            continue

        ExtractTableData().execute(table_name, out_file)


def insert(tables: list[dict[str, str]], out_database: str) -> None:
    """create and insert tables from tables description"""
    existing_tables = get_tables()
    for table in tables:
        table_name = table["table_name"]
        file_name = Path(table["output_path"]) / (table_name + ".zst")

        if not file_name.exists():
            raise RuntimeError(
                f"Trying to insert:: {file_name}; but file does not exists"
            )

        table_type = table["table_type"]

        if table in existing_tables:
            logger.warning(
                f"Skipping {file_name=} file inserstion as {table_name=} table already exists"
            )
        else:
            logger.info(
                f"Inserting file {file_name=} into {table_name=} of type {table_type=}"
            )

        # create table
        with ClickHouseClient(**ch_settings.clickhouse) as client:
            match table_type:

                case TableTypes.DNS:
                    CreateDNSMappingTable().execute(
                        client, table_name, out_db=out_database
                    )
                case TableTypes.Ping:
                    CreatePingTable().execute(client, table_name, out_db=out_database)
                case TableTypes.Traceroute:
                    CreateTracerouteTable().execute(
                        client, table_name, out_db=out_database
                    )
                case TableTypes.Score:
                    CreateScoreTable().execute(client, table_name, out_db=out_database)
                case TableTypes.VPs:
                    CreateVPsTable().execute(client, table_name, out_db=out_database)
                case TableTypes.Geoloc:
                    CreateGeolocTable().execute(client, table_name, out_db=out_database)
                case TableTypes.NS:
                    CreateNameServerTable().execute(
                        client, table_name, out_db=out_database
                    )
                case TableTypes.Schedule:
                    CreateScheduleTable().execute(
                        client, table_name, out_db=out_database
                    )

        # insert data
        cmd = f"clickhouse client --user {ch_settings.CLICKHOUSE_USERNAME}"
        cmd += f" --password {ch_settings.CLICKHOUSE_PASSWORD}"
        cmd += f" --query=\"INSERT INTO {out_database}.{table_name} FROM INFILE '{file_name}' FORMAT Native\""

        ps = subprocess.run(
            cmd,
            capture_output=True,
            shell=True,
            text=True,
        )

        logger.debug(f"{cmd=}; {ps.stdout=}; {ps.stderr=}")


def main() -> None:
    """entry point, either extract data and output to file or create and insert"""

    # create archive dir
    path_settings.ARCHIVE_PATH.mkdir(exist_ok=True, parents=True)

    # tables/output files to extract/insert
    tables = [
        {
            "table_name": "local_demo_ecs",
            "table_type": TableTypes.DNS,
            "output_path": path_settings.ARCHIVE_PATH,
        },
        # {
        #     "table_name": "vps_ecs_mapping",
        #     "table_type": TableTypes.DNS,
        #     "output_path": path_settings.ARCHIVE_PATH,
        # },
        # {
        #     "table_name": "vps_raw",
        #     "table_type": TableTypes.VPs,
        #     "output_path": path_settings.ARCHIVE_PATH,
        # },
        # {
        #     "table_name": "vps_filtered_final",
        #     "table_type": TableTypes.VPs,
        #     "output_path": path_settings.ARCHIVE_PATH,
        # },
        # {
        #     "table_name": "vps_meshed_pings",
        #     "table_type": TableTypes.Ping,
        #     "output_path": path_settings.ARCHIVE_PATH,
        # },
        # {
        #     "table_name": "vps_meshed_traceroutes",
        #     "table_type": TableTypes.Traceroute,
        #     "output_path": path_settings.ARCHIVE_PATH,
        # },
        # {
        #     "table_name": "itdk_ping",
        #     "table_type": TableTypes.Ping,
        #     "output_path": path_settings.ARCHIVE_PATH,
        # }
    ]

    insert(tables, out_database=OUT_DB)

    logger.info("GeoResolver Successfully")


if __name__ == "__main__":
    typer.run(main())
