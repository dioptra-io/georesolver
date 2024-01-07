"""general tool settings"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from common.credentials import get_ripe_atlas_credentials, get_vela_credentials


class ConstantSettings(BaseSettings):
    """all constant used in the project"""

    SPEED_OF_LIGHT: int = 300000
    SPEED_OF_INTERNET: float = SPEED_OF_LIGHT * 2 / 3


class VELASettings(BaseSettings):
    BASE_URL: str = "https://vela.caida.org/api/"
    TIMEOUT: int = 10
    CREDENTIALS: dict = get_vela_credentials()


class MeasurementSettings(BaseSettings):
    """measurement parameters"""

    # default ripe atlas parameters
    MAX_NUMBER_VP: int = 1000
    MAX_CONSECUTIVE_MEASUREMENTS: int = 99
    PING_DEFAULT_NUMBER_PACKET: int = 3

    CREDENTIALS: dict = get_ripe_atlas_credentials()


class PathSettings(BaseSettings):
    """define all path, easy to import"""

    # Default path
    DEFAULT_PATH: Path = Path(__file__).resolve().parent

    # main dataset dirs
    DATASET_PATH: Path = DEFAULT_PATH / "../datasets/"
    RIPE_DATASET: Path = DATASET_PATH / "ripe/"
    HOSTNAME_DATASET: Path = DATASET_PATH / "hostnames/"
    RESULTS_PATH: Path = DEFAULT_PATH / "../results"
    FIGURE_PATH: Path = DEFAULT_PATH / "../figures"
    TMP_PATH: Path = DEFAULT_PATH / "../tmp/"
    LOG_PATH: Path = DEFAULT_PATH / "../logs"
    ZDNS_PATH: Path = DEFAULT_PATH / "../zdns"

    # RIPE
    ANCHORS: Path = RIPE_DATASET / "anchors.json"
    PROBES: Path = RIPE_DATASET / "probes.json"
    UNFILTERED_TARGETS: Path = RIPE_DATASET / "unfiltered_targets.json"
    UNFILTERED_VPS: Path = RIPE_DATASET / "unfiltered_vps.json"
    TARGETS: Path = RIPE_DATASET / "targets.json"
    VPS: Path = RIPE_DATASET / "vps.json"
    REMOVED_VPS: Path = RIPE_DATASET / "removed_vps.json"
    TARGET_TO_VPS_DISTANCE: Path = RIPE_DATASET / "target_to_vps_distance.json"
    VPS_DISTANCE_MATRIX: Path = RIPE_DATASET / "vps_distance_matrix.json"

    # HOSTNAMES
    HOSTNAME_RAW_LINKS: Path = HOSTNAME_DATASET / "raw_links_current.csv"
    HOSTNAME_RAW: Path = HOSTNAME_DATASET / "raw_hostnames.csv"
    HOSTNAMES_ECS: Path = HOSTNAME_DATASET / "ecs_hostnames.json"
    HOSTNAMES_ANYCAST: Path = HOSTNAME_DATASET / "anycast_hostnames.json"
    HOSTNAMES_UNICAST: Path = HOSTNAME_DATASET / "unicast_hostnames.json"
    HOSTNAMES_FILTERED: Path = (
        HOSTNAME_DATASET / "filtered_hostnames.csv"
    )  # unicast + ecs

    # main datasets
    HOSTNAMES: Path = HOSTNAME_DATASET / "hostnames.csv"
    UNRESERVED_SUBNET_FILE: Path = DATASET_PATH / "all_24_subnets.csv"

    # prefix metadata
    BGP_ANNOUNCED_SUBNET: Path = DATASET_PATH / "bgp_announced_subnet.csv"
    FRONTEND_BGP_PREFIXES: Path = DATASET_PATH / "frontend_bgp_prefix.json"
    ANYCAST_PREFIXES: Path = DATASET_PATH / "anycatch-v4-prefixes.csv"
    RTT_CLUSTERING: Path = (
        DATASET_PATH
        / "clustering/rtt/cluster_per_ip_per_asn_rerun_min_sample_2_steepness_0.1.jsonl"
    )
    RTTL_CLUSTERING: Path = (
        DATASET_PATH
        / "clustering/rttl/cluster_per_ip_per_asn_rerun_min_sample_2_steepness_0.1.jsonl"
    )
    IP_INFO_CLUSTERING: Path = DATASET_PATH / "clustering/ips.jsonl"

    # CDN
    CDN_DATASET_PATH: Path = DATASET_PATH / "cdn"
    GOOGLE_POP_DATASET: Path = CDN_DATASET_PATH / "Google_GGC_02-08-2023.geojson"

    # DNS mapping dirs
    DNS_MAPPING_TARGETS: Path = DATASET_PATH / "dns_mapping_targets.json"
    DNS_MAPPING_VPS: Path = DATASET_PATH / "dns_mapping_vps.json"
    DNS_MAPPING_ALL_SUBNETS: Path = DATASET_PATH / "dns_mapping_all.json"
    DNS_MAPPING_BGP_SUBNETS: Path = DATASET_PATH / "dns_mapping_bgp.json"

    # measurements
    MEASUREMENT_PATH: Path = DEFAULT_PATH / "../measurements"
    MEASUREMENT_CONFIG_PATH: Path = DEFAULT_PATH / "../measurements/config"
    MEASUREMENT_RESULTS_PATH: Path = DEFAULT_PATH / "../measurements/results"

    # static datasets
    STATIC_DATASET_PATH: Path = DATASET_PATH / "static/"
    COUNTRIES_DEFAULT_LOCATION: Path = STATIC_DATASET_PATH / "countries.txt"
    RESPONSIVE_ADDRESSES_PER_SUBNET: Path = (
        STATIC_DATASET_PATH / "responsive_addresses_per_subnet.fsdb"
    )
    ALL_RESPONSIVE_ADDRESSES: Path = STATIC_DATASET_PATH / "all_responsive_addr.json"

    TARGETS_PER_SUBNET: Path = STATIC_DATASET_PATH / "targets_per_subnet.json"


class ClickhouseSettings(BaseSettings):
    """general settings, credentials"""

    # workaround to override LD_LIBRARY_PATH. Otherwise cannot insert into clickhouse
    load_dotenv(override=True)

    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_USERNAME: str = "default"
    CLICKHOUSE_PASSWORD: str = ""
    CLICKHOUSE_DB: str = "dns_geoloc"

    # dns resolution
    DNS_MAPPING_ANCHOR_TABLE: str = "dns_mapping_anchor"
    DNS_MAPPING_PROBE_TABLE: str = "dns_mapping_probe"
    DNS_MAPPING_TARGETS_TABLE: str = "dns_mapping_targets"
    DNS_MAPPING_VPS_TABLE: str = "dns_mapping_vps"
    DNS_MAPPING_ALL_SUBNETS_TABLE: str = "dns_mapping_all_subnets"
    DNS_MAPPING_PERIODIC_TABLE: str = "dns_mapping_periodic"
    DNS_MAPPING_WITH_METADATA_TABLE: str = "dns_mapping_with_metadata_table"

    DNS_MAPPING_TARGETS_TABLE: str = "dns_mapping_targets"
    DNS_MAPPING_VPS_TABLE: str = "dns_mapping_vps"

    # measurements
    PING_ANCHOR_TO_POP_TABLE: str = "ping_anchor_to_pop"
    PING_PROBES_MESHED_ANCHOR: str = "ping_meshed_probes_to_anchor"
    PING_PROBES_MESHED_SUBNET: str = "ping_meshed_probes_to_subnet"
    PING_PROBE_TO_POP_TABLE: str = "ping_probe_to_pop"
    PING_VPS_TO_TARGETS: str = "ping_vps_to_targets"
    PING_VPS_TO_SUBNET: str = "ping_vps_to_subnet"
    PING_VPS_TO_POP_TABLE: str = "ping_vps_to_pop"
    PING_TARGET_TO_POP_TABLE: str = "ping_vps_to_pop"

    # geolocation
    POP_GEOLOCATION: str = "pop_geolocation"
    ALL_POP_GEOLOCATION: str = "all_pop_geolocation"

    @property
    def clickhouse(self):
        return {
            "host": self.CLICKHOUSE_HOST,
            "username": self.CLICKHOUSE_USERNAME,
            "password": self.CLICKHOUSE_PASSWORD,
            "db": self.CLICKHOUSE_DB,
        }
