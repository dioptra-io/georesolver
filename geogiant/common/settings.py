"""general tool settings"""
from pathlib import Path
from dotenv import load_dotenv

from pydantic_settings import BaseSettings


class ConstantSettings(BaseSettings):
    """all constant used in the project"""

    SPEED_OF_LIGHT: int = 300000
    SPEED_OF_INTERNET: float = SPEED_OF_LIGHT * 2 / 3


class PathSettings(BaseSettings):
    """define main project directories and static files"""

    # Default path
    DEFAULT: Path = Path(__file__).resolve().parent

    # main dataset dirs
    DATASET: Path = DEFAULT / "../datasets/"
    RESULTS_PATH: Path = DEFAULT / "../results"
    FIGURE_PATH: Path = DEFAULT / "../figures"
    TMP_PATH: Path = DEFAULT / "../tmp/"
    LOG_PATH: Path = DEFAULT / "../logs"

    # hostnames default input files
    HOSTNAMES_MILLIONS: Path = DATASET / "raw_links_current.csv"
    CDN_HOSTNAMES_RAW: Path = DATASET / "cdn_hostnames_raw.csv"

    # bgp route views data
    RIB_TABLE: Path = DATASET / "rib_table.dat"

    # anycatch database
    ANYCATCH_DATA: Path = DATASET / "anycatch-v4-prefixes.csv"

    # CDNs datasets
    GOOGLE_POP_GEO_DATA: Path = DATASET / "Google_GGC_02-08-2023.geojson"

    # default location countries
    COUNTRIES_DEFAULT_GEO: Path = DATASET / "countries_default_geo.txt"
    COUNTRIES_CONTINENT: Path = DATASET / "countries_continent.csv"

    # Verpoelofter
    VERPOELOFTER: Path = DATASET / "responsive_addresses_per_subnet.fsdb"


class ClickhouseSettings(BaseSettings):
    """general settings, credentials"""

    # workaround to override LD_LIBRARY_PATH. Otherwise cannot insert into clickhouse
    load_dotenv(override=True)

    # Clickhouse driver parameters
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_USERNAME: str = "default"
    CLICKHOUSE_PASSWORD: str = ""
    CLICKHOUSE_DB: str = "dns_geoloc"

    # ZDNS tables
    DNS_MAPPING_TARGETS: str = "raw_dns_mapping_targets"
    DNS_MAPPING_VPS: str = "dns_mapping_vps"
    DNS_MAPPING_VPS_RAW: str = "raw_dns_mapping_vps"
    DNS_MAPPING_METADATA_TARGETS: str = "dns_mapping_metadata_targets"
    DNS_MAPPING_METADATA_VPS: str = "dns_mapping_metadata_vps"

    # RIPE Atlas tables
    TARGETS_TABLE_RAW: str = "targets_raw"
    TARGETS_TABLE: str = "targets"
    VPS_RAW: str = "vps_raw"
    VPS: str = "vps"
    PING_VPS_TO_TARGET: str = "ping_vps_to_target"
    PING_VPS_TO_FRONTEND: str = "ping_vps_to_frontend"

    # Geoloc tables
    TARGET_GEOLOCATION: str = "target_geolocation"
    POP_GEOLOCATION: str = "pop_geolocation"

    # TODO: create ping measurement models and
    # TODO: create table with uuid?

    @property
    def clickhouse(self):
        return {
            "host": self.CLICKHOUSE_HOST,
            "username": self.CLICKHOUSE_USERNAME,
            "password": self.CLICKHOUSE_PASSWORD,
            "db": self.CLICKHOUSE_DB,
        }


class RIPEAtlasSettings(BaseSettings):
    """RIPE Atlas module settings"""

    # Default path
    DEFAULT: Path = Path(__file__).resolve().parent

    # credentials
    USERNAME: str = ""
    SECRET_KEY: str = ""
    IP_VERSION: int = 4

    # default ripe atlas parameters
    MAX_VP: int = 1000
    MAX_MEASUREMENT: int = 99
    PING_NB_PACKETS: int = 3

    # debugging
    MEASUREMENTS_CONFIG: Path = DEFAULT / "../measurements/config"

    BASE_URL: str = "https://atlas.ripe.net/api/v2"
    KEY_URL: str = f"?key={SECRET_KEY}"

    MEASUREMENT_URL: str = f"{BASE_URL}/measurements/{KEY_URL}"


class ZDNSSettings(PathSettings, ClickhouseSettings):
    """ZNDS module settings"""

    # ZDNS binary path
    DEFAULT_PATH: Path = Path(__file__).resolve().parent
    EXEC_PATH: Path = DEFAULT_PATH / "../zdns/zdns_binary"

    # ZDNS tool parameters
    RECORD_TYPE: str = "A"
