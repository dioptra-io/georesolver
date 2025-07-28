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
        # Ping tables
        "itdk_ping",
        "vps_meshed_pings",
        "vps_meshed_pings_ipv6",
        "meshed_cdns_pings",
        "single_radius_ping",
        "single_radius_georesolver_ping",
        "vps_meshed_pings_CoNEXT_summer_submision",
        # VPs tables
        "vps_raw",
        "vps_filtered",
        "vps_filtered_final",
        "vps_filtered_ipv6",
        "vps_filtered_final_CoNEXT_winter_submision"
        # DNS tables
        "vps_ecs_mapping",
        "vps_ecs_mapping__2025_04_13",
        "vps_ecs_mapping_ecs_ipv6_latest",
        "meshed_cdns_ecs",
        # Traceroute tables
        "vps_meshed_traceroutes",
        "vps_meshed_traceroutes_ipv6",
    ]

    for table_name in tables:
        out_file: Path = path_settings.FTP_FILE_PATH / (table_name + ".zst")
        logger.info(f"{table_name=}; {out_file=}")

        if out_file.exists():
            logger.warning(f"File {out_file=} already exists")
            continue

        ExtractTableData().execute(table_name, out_file)

    logger.info("GeoResolver Successfully")

    # TODO: export files into datasets dir in FTP


if __name__ == "__main__":
    main()
