"""general tool settings"""

from pathlib import Path
from dotenv import load_dotenv
from typing import Optional


from pydantic_settings import BaseSettings


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
    DATASET: Path = DEFAULT / "../datasets/"
    RESULTS_PATH: Path = DEFAULT / "../results"
    FIGURE_PATH: Path = DEFAULT / "../figures"
    TMP_PATH: Path = DEFAULT / "../tmp/"
    LOG_PATH: Path = DEFAULT / "../logs"
    CONFIG_PATH: Path = DEFAULT / "../config"
    RIPE_ATLAS_PUBLIC_MEASUREMENTS: Path = DATASET / "ripe_atlas_public_measurements"
    RIPE_ATLAS_PUBLIC_PINGS: Path = DATASET / "ripe_atlas_public_pings"

    # internet scale file
    ADDRESS_FILE: Path = (
        DATASET
        / "internet_address_hitlist_it106w-20231222/internet_address_hitlist_it106w-20231222.fsdb"
    )

    USER_HITLIST_FILE: Path = DATASET / "ipv4_hitlist.json"

    # evaluation specific datasets
    END_TO_END_DATASET: Path = DATASET / "end_to_end_evaluation"
    INTERNET_SCALE_DATASET: Path = DATASET / "internet_scale_evaluation"
    INTERNET_SCALE_RESULTS: Path = RESULTS_PATH / "internet_scale_evaluation"

    # hostnames default input files
    HOSTNAMES_MILLIONS: Path = DATASET / "hostnames_1M.csv"
    HOSTNAMES_CDN: Path = DATASET / "hostnames_cdn.csv"
    HOSTNAMES_ECS: Path = DATASET / "hostnames_ecs.csv"

    # bgp route views data
    RIB_TABLE: Path = DATASET / "rib_table.dat"

    # anycatch database
    ANYCATCH_DATA: Path = DATASET / "anycatch-v4-prefixes.csv"

    # CDNs datasets
    GOOGLE_POP_GEO_DATA: Path = DATASET / "Google_GGC_02-08-2023.geojson"

    # default location countries
    COUNTRIES_INFO: Path = DATASET / "countries_info.csv"
    COUNTRIES_DEFAULT_GEO: Path = DATASET / "country_codes.csv"
    COUNTRIES_CONTINENT: Path = DATASET / "countries_continent.csv"

    # Verpoelofter
    VERPOELOFTER: Path = DATASET / "responsive_addresses_per_subnet.fsdb"

    # IP INFO
    IP_INFO_CLUSTERING: Path = DATASET / "ips.jsonl"

    # Others
    VPS_PAIRWISE_DISTANCE: Path = DATASET / "vps_pairwise_distance.json"
    GREEDY_VPS: Path = DATASET / "greedy_vps.json"
    REMOVED_VPS: Path = DATASET / "removed_vps.json"

    # ECS-DNS old data
    OLD_TARGETS: Path = DATASET / "old/ripe/targets.json"
    OLD_VPS: Path = DATASET / "old/ripe/vps.json"
    OLD_VPS_PAIRWISE_DISTANCE: Path = (
        DATASET / "old/ripe/old_vps_pairwise_distance.json"
    )
    OLD_DATA_PATH: Path = Path("/storage/hugo/ecs-dns-data-old/")
    OLD_DNS_MAPPING_DATA: Path = OLD_DATA_PATH / "dns_mapping.zst"
    OLD_PING_VPS_TO_TARGET_DATA: Path = OLD_DATA_PATH / "ping_vps_to_targets.zst"
    OLD_PING_VPS_TO_SUBNET_DATA: Path = OLD_DATA_PATH / "ping_vps_to_subnet.zst"

    VPS_RAW_DATA: Path = OLD_DATA_PATH / "vps_raw.zst"
    DNS_MAPPING_VPS_RAW_DATA: Path = OLD_DATA_PATH / "raw_dns_mapping_vps.zst"

    # GeoResolver settings
    DEFAULT_HOSTNAME_FILE: Path = DATASET / "internet_scale_hostnames.csv"


class ClickhouseSettings(BaseSettings):
    """general settings, credentials"""

    # workaround to override LD_LIBRARY_PATH. Otherwise cannot insert into clickhouse
    load_dotenv(override=True)

    # Clickhouse driver parameters
    BASE_URL: str = "http://localhost:8123"
    DATABASE_OLD: str = "geogiant"
    DATABASE: str = "geogiant"
    DATABASE: str = "imc2024"

    USERNAME: str = "default"
    PASSWORD: str = ""

    # ZDNS tables
    DNS_MAPPING_TARGETS: str = "raw_dns_mapping_targets"
    DNS_MAPPING_VPS: str = "vps_mapping_ecs_selection"
    DNS_MAPPING_VPS_RAW: str = "raw_dns_mapping_vps"
    DNS_MAPPING_ECS: str = "ecs_dns_mapping"
    DNS_MAPPING_METADATA_TARGETS: str = "dns_mapping_metadata_targets"
    DNS_MAPPING_METADATA_VPS: str = "dns_mapping_metadata_vps"

    # RIPE Atlas tables
    TARGETS_TABLE_RAW: str = "targets_raw"
    TARGETS_TABLE: str = "targets"
    VPS_RAW: str = "vps_raw"
    VPS_FILTERED: str = "filtered_vps"
    VPS: str = "vps"
    PING_VPS_TO_TARGET: str = "ping_vps_to_targets"
    PING_VPS_TO_FRONTEND: str = "ping_vps_to_frontend"

    # Traceroute validation
    TRACEROUTES_LAST_MILE_DELAY: str = "traceroutes_last_mile_delay"

    # Geoloc tables
    TARGET_GEOLOCATION: str = "target_geolocation"
    POP_GEOLOCATION: str = "pop_geolocation"

    # Public RIPE Atlas data
    RIPE_ATLAS_TRACEROUTES: str = "ripe_atlas_traceroute"
    RIPE_ATLAS_TRACEROUTE_GEOLOC: str = "ripe_atlas_traceroute_geoloc"

    # ECS-DNS old tables
    OLD_DNS_MAPPING: str = "old_dns_mapping"
    OLD_DNS_MAPPING_WITH_METADATA: str = "old_dns_mapping_with_metadata"
    OLD_PING_VPS_TO_TARGET: str = "old_ping_vps_to_target"
    OLD_PING_VPS_TO_SUBNET: str = "old_ping_vps_to_subnet"

    @property
    def clickhouse(self):
        return {
            "base_url": self.BASE_URL,
            "database": self.DATABASE,
            "username": self.USERNAME,
            "password": self.PASSWORD,
            "settings": {
                "max_query_size": 1000000000000000,
            },
        }

    # GeoResolver settings
    ECS_TARGET_TABLE: str = "ecs_target_table"
    ECS_VPS_TABLE: str = "vps_mapping_ecs"
    SCORE_TARGET_TABLE: str = "score_target_table"
    PING_TARGET_TABLE: str = "ping_target_table"
    GEOLOC_TARGET_TABLE: str = "geoloc_target_table"

    VPS_FILTERED_TABLE: str = "filtered_vps"


class RIPEAtlasSettings(PathSettings, ClickhouseSettings):
    """RIPE Atlas module settings"""

    # Default path
    DEFAULT: Path = Path(__file__).resolve().parent

    # credentials
    RIPE_ATLAS_USERNAME: str = ""
    RIPE_ATLAS_SECRET_KEY: str = ""
    IP_VERSION: int = 4

    # default ripe atlas parameters
    MAX_VP: int = 1_000
    MAX_MEASUREMENT: int = 2_000
    PING_NB_PACKETS: int = 3
    PROTOCOL: str = "ICMP"

    # debugging
    MEASUREMENTS_CONFIG: Path = DEFAULT / "../measurements/config"
    MEASUREMENTS_SCHEDULE: Path = DEFAULT / "../measurements/schedule"

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
