"""general tool settings"""

import sys

from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

from pydantic_settings import BaseSettings

load_dotenv(override=True)


class ConstantSettings(BaseSettings):
    """all constant used in the project"""

    SPEED_OF_LIGHT: int = 300000
    SPEED_OF_INTERNET: float = SPEED_OF_LIGHT * 2 / 3
    PROBING_BUDGET: int = 50


class PathSettings(BaseSettings):
    """define main project directories and static files"""

    # Default path
    DEFAULT: Path = Path(__file__).resolve().parent

    # deployment
    GITHUB_TOKEN: str = ""
    DOCKER_USERNAME: str = ""
    SSH_USER: str = ""
    SUDO_PWD: str = ""

    # main dataset dirs
    TMP_PATH: Path = DEFAULT / "../tmp/"
    LOG_PATH: Path = DEFAULT / "../logs"
    DATASET: Path = DEFAULT / "../datasets/"
    CONFIG_PATH: Path = DEFAULT / "../config"
    FIGURE_PATH: Path = DEFAULT / "../figures"
    ARCHIVE_PATH: Path = DEFAULT / "../archive"
    RESULTS_PATH: Path = DEFAULT / "../results"
    USER_DATASETS: Path = DEFAULT / "../user_datasets"
    EXPERIMENT_PATH: Path = DEFAULT / "../experiments"

    # Static files from other organization
    STATIC_FILES: Path = DATASET / "static_files/"
    RIB_TABLE: Path = STATIC_FILES / "rib_table.dat"
    USER_HITLIST_FILE: Path = STATIC_FILES / "ipv4_hitlist.json"
    ANYCATCH_DATA: Path = STATIC_FILES / "anycatch-v4-prefixes.csv"
    VERPOELOFTER: Path = STATIC_FILES / "responsive_addresses_per_subnet.fsdb"
    ADDRESS_FILE: Path = STATIC_FILES / "internet_address_hitlist_it106w-20231222.fsdb"

    # hostnames default input files
    HOSTNAME_FILES: Path = DATASET / "hostname_files/"
    HOSTNAMES_CDN: Path = HOSTNAME_FILES / "hostnames_cdn.csv"
    HOSTNAMES_ECS: Path = HOSTNAME_FILES / "hostnames_ecs.csv"
    HOSTNAMES_MILLIONS: Path = HOSTNAME_FILES / "hostnames_1M.csv"
    HOSTNAMES_GEORESOLVER: Path = HOSTNAME_FILES / "hostnames_georesolver.csv"

    # default location countries
    COUNTRY_FILES: Path = DATASET / "country_files/"
    COUNTRIES_INFO: Path = COUNTRY_FILES / "countries_info.csv"
    COUNTRIES_DEFAULT_GEO: Path = COUNTRY_FILES / "country_codes.csv"
    COUNTRIES_CONTINENT: Path = COUNTRY_FILES / "countries_continent.csv"
    COUNTRIES_GOOGLE_POP_GEO: Path = COUNTRY_FILES / "Google_GGC_02-08-2023.geojson"

    # evaluation specific datasets
    END_TO_END_DATASET: Path = DATASET / "end_to_end_evaluation"
    INTERNET_SCALE_DATASET: Path = DATASET / "internet_scale_evaluation"
    INTERNET_SCALE_RESULTS: Path = RESULTS_PATH / "internet_scale_evaluation"

    # measurement logging
    MEASUREMENTS_CONFIG: Path = DEFAULT / "../measurements/config"
    MEASUREMENTS_SCHEDULE: Path = DEFAULT / "../measurements/schedule"

    # IP INFO
    IP_INFO_FILES: Path = DATASET / "ip_info/"
    IP_INFO_CLUSTERING: Path = IP_INFO_FILES / "ips.jsonl"

    # Generated files
    GREEDY_VPS: Path = DATASET / "greedy_vps.json"
    REMOVED_VPS: Path = DATASET / "removed_vps.json"
    VPS_PAIRWISE_DISTANCE: Path = DATASET / "vps_pairwise_distance.json"


class ClickhouseSettings(BaseSettings):
    """general settings, credentials"""

    # Clickhouse driver parameters
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_DATABASE: str = "GeoResolver"
    CLICKHOUSE_DATABASE_EVAL: str = "GeoResolver_evaluation"

    CLICKHOUSE_USERNAME: str = "default"
    CLICKHOUSE_PASSWORD: str = ""

    # VPs related tables
    VPS_RAW_TABLE: str = "vps_raw"
    VPS_ALL_TABLE: str = "vps_all"
    VPS_FILTERED_TABLE: str = "vps_filtered"
    VPS_FILTERED_FINAL_TABLE: str = "vps_filtered_final"
    VPS_MESHED_PINGS_TABLE: str = "vps_meshed_pings"
    VPS_MESHED_TRACEROUTE_TABLE: str = "vps_meshed_traceroutes"
    VPS_ECS_MAPPING_TABLE: str = "vps_ecs_mapping"

    # Target related tables
    TARGET_ECS_MAPPING_TABLE: str = "target_ecs_mapping"
    TARGET_SCORE_TABLE: str = "target_score"
    TARGET_PING_TABLE: str = "target_ping"
    TARGET_GEOLOC_TABLE: str = "target_geoloc"

    # TODO: hostname related tables

    @property
    def clickhouse(self):
        return {
            "base_url": f"http://{self.CLICKHOUSE_HOST}:{self.CLICKHOUSE_PORT}",
            "database": self.CLICKHOUSE_DATABASE,
            "username": self.CLICKHOUSE_USERNAME,
            "password": self.CLICKHOUSE_PASSWORD,
            "settings": {
                "max_query_size": 1000000000000000,
            },
        }


class RIPEAtlasSettings(PathSettings, ClickhouseSettings):
    """RIPE Atlas module settings"""

    # credentials
    RIPE_ATLAS_USERNAME: str = ""
    RIPE_ATLAS_SECRET_KEY: str = ""
    RIPE_ATLAS_SECRET_KEY_SECONDARY: str = ""
    IP_VERSION: int = 4

    # RIPE Atlas parameters
    MAX_VP: int = 1_000
    MAX_MEASUREMENT: int = 2_000
    PING_NB_PACKETS: int = 3
    PROTOCOL: str = "ICMP"

    API_URL: str = "https://atlas.ripe.net/api/v2"
    KEY_URL: str = f"?key={RIPE_ATLAS_SECRET_KEY}"
    MEASUREMENT_URL: str = f"{API_URL}/measurements/{KEY_URL}"


class ZDNSSettings(PathSettings, ClickhouseSettings):
    """ZNDS module settings"""

    # ZDNS binary path
    DEFAULT_PATH: Path = Path(__file__).resolve().parent
    EXEC_PATH: Path = DEFAULT_PATH / "../zdns/zdns_binary"

    # ZDNS tool parameters
    RECORD_TYPE: str = "A"


def setup_logger(
    log_file_path: Path,
    verbose: bool = False,
    to_stdout: bool = False,
) -> None:
    """setup logging with output file and stdout if required, define log level"""
    logger.remove()

    # get log level
    if verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    logger.add(log_file_path, level=log_level)

    # in case we want to keep logs to stdout
    if to_stdout:
        logger.add(sys.stdout, level=log_level)
