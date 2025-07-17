"""utility function for GeoResolver"""

from pyasn import pyasn
from collections import defaultdict

from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_parsed_vps(vps: list, asndb: pyasn, removed_vps: list = []) -> dict:
    """parse vps list to a dict for fast retrieval. Keys depends on granularity"""
    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    vps_bgp_prefix = defaultdict(list)

    for vp in vps:
        if vp["addr"] in removed_vps:
            continue
        vp_addr = vp["addr"]
        subnet = get_prefix_from_ip(vp_addr)
        vp_asn, vp_bgp_prefix = route_view_bgp_prefix(vp_addr, asndb)
        vp_lat, vp_lon = vp["lat"], vp["lon"]
        vp_country_code = vp["country_code"]

        vps_subnet[subnet].append(vp_addr)
        vps_bgp_prefix[vp_bgp_prefix].append(vp_addr)
        vps_coordinates[vp_addr] = (vp_lat, vp_lon, vp_country_code, vp_asn)

    return vps_subnet, vps_coordinates


def get_vps_per_subnet(vps: list) -> dict:
    """group vps per subnet"""
    vps_subnet = defaultdict(list)

    for vp in vps:
        vps_subnet[vp["subnet"]].append(vp)

    return vps_subnet
