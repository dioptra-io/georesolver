"""general tool settings"""
from pathlib import Path

from dotenv import load_dotenv
from common_settings import CommonSettings


class RIPEAtlasSettings(CommonSettings):
    """general settings, credentials"""

    # workaround to override LD_LIBRARY_PATH. Otherwise cannot insert into clickhouse
    load_dotenv(override=True)

    # credentials
    username: str = ""
    secret_key: str = ""
    ip_version: int = 4

    # urls
    base_url: str = "https://atlas.ripe.net/api/v2"
    key_url: str = f"?key={secret_key}"
    measurement_url: str = f"{base_url}/measurements/{key_url}"

    # default ripe atlas parameters
    max_vp: int = 1000
    max_measurement: int = 99
    ping_nb_packets: int = 3


class ZDNSSettings(CommonSettings):
    """ZNDS settings"""

    ZDNS_PATH: Path = self.DEFAULT_PATH / "../zdns"

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
    def credentials(self):
        return {
            "base_url": self.base_url,
            "username": self.username,
            "secret_key": self.secret_key,
            "password": self.CLICKHOUSE_PASSWORD,
            "db": self.CLICKHOUSE_DB,
        }
