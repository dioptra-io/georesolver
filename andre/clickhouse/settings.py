"""Clickhouse settings (credentials, tables, database, etc.)"""
from pathlib import Path

from dotenv import load_dotenv
from common.settings import CommonSettings

class ClickhouseSettings(CommonSettings):
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
