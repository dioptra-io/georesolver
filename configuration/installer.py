"""extract, insert data into the right tables for evaluation and measurements"""

import typer
import subprocess

from enum import Enum
from pathlib import Path
from loguru import logger
from pych_client import ClickHouseClient

from georesolver.clickhouse import (
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
from georesolver.common.files_utils import download_byte_file
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


def download_from_ftp(file_name: str, output_path: Path) -> None:
    """upload_to_ftp data from an input list of tables/output files"""

    url = "ftp://132.227.123.74/CoNEXT_artifacts/" + file_name

    if (output_path / file_name).exists():
        logger.warning(
            f"File {file_name} already exists at {output_path}, skipping download"
        )
        return

    download_byte_file(
        url=url,
        output_path=output_path / file_name,
        username=path_settings.FTP_USERNAME,
        password=path_settings.FTP_PASSWORD,
    )


def insert_clickhouse(
    table: dict[str, str], input_path: Path, out_database: str
) -> None:
    """create and insert tables from tables description"""
    table_name = table["table_name"]
    table_type = table["table_type"]
    file_name = input_path / (table_name + ".zst")

    if not file_name.exists():
        raise RuntimeError(f"Trying to insert:: {file_name}; but file does not exists")

    if table_name in get_tables():
        logger.warning(
            f"Skipping {file_name=} file inserstion as {table_name=} table already exists"
        )
        return
    else:
        logger.info(
            f"Inserting file {file_name=} into {table_name=} of type {table_type=}"
        )

    # create table
    with ClickHouseClient(**ch_settings.clickhouse) as client:
        match table_type:

            case TableTypes.DNS:
                CreateDNSMappingTable().execute(client, table_name, out_db=out_database)
            case TableTypes.Ping:
                CreatePingTable().execute(client, table_name, out_db=out_database)
            case TableTypes.Traceroute:
                CreateTracerouteTable().execute(client, table_name, out_db=out_database)
            case TableTypes.Score:
                CreateScoreTable().execute(client, table_name, out_db=out_database)
            case TableTypes.VPs:
                CreateVPsTable().execute(client, table_name, out_db=out_database)
            case TableTypes.Geoloc:
                CreateGeolocTable().execute(client, table_name, out_db=out_database)
            case TableTypes.NS:
                CreateNameServerTable().execute(client, table_name, out_db=out_database)
            case TableTypes.Schedule:
                CreateScheduleTable().execute(client, table_name, out_db=out_database)

    # insert data
    cmd = f"clickhouse-client --user {ch_settings.CLICKHOUSE_USERNAME}"
    cmd += (
        f" --password {ch_settings.CLICKHOUSE_PASSWORD}"
        if ch_settings.CLICKHOUSE_PASSWORD
        else ""
    )
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
        # DNS Tables
        {
            "table_name": "vps_ecs_mapping__2025_04_13",
            "table_type": TableTypes.DNS,
        },
        {
            "table_name": "vps_ecs_mapping_ecs_ipv6_latest",
            "table_type": TableTypes.DNS,
        },
        {
            "table_name": "meshed_cdns_ecs",
            "table_type": TableTypes.DNS,
        },
        {
            "table_name": "vps_ecs_mapping",
            "table_type": TableTypes.DNS,
        },
        # VPs tables
        {
            "table_name": "vps_raw",
            "table_type": TableTypes.VPs,
        },
        {
            "table_name": "vps_filtered",
            "table_type": TableTypes.VPs,
        },
        {
            "table_name": "vps_filtered_final",
            "table_type": TableTypes.VPs,
        },
        {
            "table_name": "vps_filtered_ipv6",
            "table_type": TableTypes.VPs,
        },
        # Ping Tables
        {
            "table_name": "vps_meshed_pings",
            "table_type": TableTypes.Ping,
        },
        {
            "table_name": "meshed_cdns_pings",
            "table_type": TableTypes.Ping,
        },
        {
            "table_name": "itdk_ping",
            "table_type": TableTypes.Ping,
        },
        {
            "table_name": "vps_meshed_pings_ipv6",
            "table_type": TableTypes.Ping,
        },
        {
            "table_name": "single_radius_ping",
            "table_type": TableTypes.Ping,
        },
        {
            "table_name": "single_radius_georesolver_ping",
            "table_type": TableTypes.Ping,
        },
        {
            "table_name": "vps_meshed_pings_CoNEXT_summer_submision",
            "table_type": TableTypes.Ping,
        },
        # Traceroute Tables
        {
            "table_name": "vps_meshed_traceroutes",
            "table_type": TableTypes.Traceroute,
        },
        {
            "table_name": "vps_meshed_traceroutes_ipv6",
            "table_type": TableTypes.Traceroute,
        },
    ]

    for table in tables:
        file_name = table["table_name"] + ".zst"
        download_from_ftp(
            file_name=file_name, output_path=path_settings.CLICKHOUSE_FILE_PATH
        )
        insert_clickhouse(
            table, input_path=path_settings.CLICKHOUSE_FILE_PATH, out_database=OUT_DB
        )

    logger.info(
        "Clickhouse data succesfully retrieved from FTP server and installed in clickhouse"
    )
    logger.info("You are ready to start using GeoResolver!!")


if __name__ == "__main__":
    typer.run(main())
