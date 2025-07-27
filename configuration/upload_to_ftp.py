"""extract data from Clickhouse GeoResolver to upload to FTP server dir"""

from pathlib import Path
from loguru import logger

from georesolver.clickhouse import ExtractTableData
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


def main() -> None:
    """upload necessary tables and files to FTP server"""

    # create archive dir
    path_settings.FTP_FILE_PATH.mkdir(exist_ok=True, parents=True)

    # tables/output files to extract/insert
    tables = [
        "meshed_cdns_pings_test",
        "vps_meshed_pings_CoNEXT_summer_submision",
        "vps_ecs_mapping__2025_04_13",
        "vps_raw",
        "vps_filtered_final",
        "vps_meshed_pings",
        "vps_meshed_traceroutes",
        "itdk_ping",
    ]

    for table in tables:
        table_name = table["table_name"]
        out_file: Path = Path(table["output_path"]) / (table_name + ".zst")
        logger.info(f"{table_name=}; {out_file=}")

        if out_file.exists():
            logger.warning(f"File {out_file=} already exists")
            continue

        ExtractTableData().execute(table_name, out_file)

    logger.info("GeoResolver Successfully")

    # TODO: export files into datasets dir in FTP


if __name__ == "__main__":
    main()
