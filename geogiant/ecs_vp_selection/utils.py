from dataclasses import dataclass
from pyasn import pyasn
from collections import defaultdict
from loguru import logger

from geogiant.common.geoloc import polygon_centroid, weighted_centroid
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.geoloc import distance


@dataclass(frozen=True)
class ResultsScore:
    client_granularity: str
    answer_granularity: str
    scores: list
    inconsistent_mappings: list


@dataclass(frozen=True)
class Score:
    answer_granularity: str
    scores: list


def select_one_vp_per_as_city(
    raw_vp_selection: list,
    vp_coordinates: dict,
    threshold: int = 40,
) -> list:
    """from a list of VP, select one per AS and per city"""
    filtered_vp_selection = []
    selected_vps_per_asn = defaultdict(list)

    for vp_addr, score in raw_vp_selection:
        vp_lat, vp_lon, vp_asn = vp_coordinates[vp_addr]

        # take at least one probe per AS
        if vp_asn not in selected_vps_per_asn:
            filtered_vp_selection.append((vp_addr, score))

            selected_vps_per_asn[vp_asn].append((vp_addr, vp_lat, vp_lon))

        else:
            # check if we already selected a VP in the same area (threshold)
            selected_close = False
            for _, selected_probe_lat, selected_probe_lon in selected_vps_per_asn[
                vp_asn
            ]:
                probe_distance = distance(
                    vp_lat, selected_probe_lat, vp_lon, selected_probe_lon
                )

                # do not select two VPs that are close together
                if probe_distance < threshold:
                    selected_close = True
                    break

            if not selected_close:
                filtered_vp_selection.append((vp_addr, score))

    return filtered_vp_selection


def get_vp_to_pops_dst(vp_lat: float, vp_lon: float, hostname_pops: dict) -> list:
    """get best pop for vp"""
    vp_to_pops_dst = []
    for pop_subnet, (pop_lat, pop_lon) in hostname_pops.items():
        if pop_lat != -1 and pop_lon != -1:
            d = distance(vp_lat, pop_lat, vp_lon, pop_lon)
            vp_to_pops_dst.append((pop_subnet, pop_lat, pop_lon, d))

    return vp_to_pops_dst


def best_ecs_target_geoloc(
    vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> tuple[float, float]:
    """simply take the VP with the highest score as target geoloc proxy"""
    for vp_addr in vps_assigned:
        try:
            vp_lat, vp_lon = vps_coordinate[vp_addr]
            break

        except KeyError:
            logger.info(f"{vp_addr} missing from ripe atlas set")

    return (vp_lat, vp_lon)


def get_ecs_pings(
    target_associated_vps: list,
    ping_to_target: list,
) -> list:
    vp_selection = []

    # filter out all vps not included by ecs-dns methodology
    for vp_addr, min_rtt in ping_to_target:
        if vp_addr in target_associated_vps:
            vp_selection.append((vp_addr, min_rtt))

    return vp_selection


def get_parsed_vps(vps: list, asndb: pyasn) -> dict:
    """parse vps list to a dict for fast retrieval. Keys depends on granularity"""
    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    vps_bgp_prefix = defaultdict(list)

    for vp in vps:
        vp_addr = vp["address_v4"]
        subnet = get_prefix_from_ip(vp_addr)
        vp_asn, vp_bgp_prefix = route_view_bgp_prefix(vp_addr, asndb)
        vp_lon, vp_lat = vp["lon"], vp["lat"]

        vps_subnet[subnet].append(vp_addr)
        vps_bgp_prefix[vp_bgp_prefix].append(vp_addr)
        vps_coordinates[vp_addr] = (vp_lat, vp_lon, vp_asn)

    return vps_subnet, vps_bgp_prefix, vps_coordinates


def ecs_target_geoloc(
    vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> tuple[float, float]:
    """
    from a list of selected vps selected with ecs-dns,
    return an estimation of the target ip address
    """
    points = []
    for vp_addr in vps_assigned:
        try:
            vp_lat, vp_lon = vps_coordinate[vp_addr]
            points.append((vp_lat, vp_lon))
        except KeyError:
            logger.info(f"{vp_addr} missing from ripe atlas set")

    if points:
        return polygon_centroid(points)

    return None, None


def get_vp_weight(
    vp_i: str, vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> float:
    """
    get vp weight for centroid calculation.
    we take w = 1 / sum(d_i_j)
    """
    sum_d = 0
    vp_i_lat, vp_i_lon = vps_coordinate[vp_i]
    for vp_j in vps_assigned:
        if vp_j == vp_i:
            continue
        try:
            vp_j_lat, vp_j_lon = vps_coordinate[vp_j]
        except KeyError:
            continue

        sum_d += distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

    if not sum_d:
        return 0

    return 1 / sum_d


def weighted_ecs_target_geoloc(
    vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> tuple[float, float]:
    """
    from a list of selected vps selected with ecs-dns,
    return an estimation of the target ip address
    """
    points = []
    for vp_addr in vps_assigned:
        try:
            vp_lat, vp_lon = vps_coordinate[vp_addr]
            vp_weight = get_vp_weight(vp_addr, vps_assigned, vps_coordinate)
            points.append((vp_lat, vp_lon, vp_weight))

        except KeyError:
            logger.info(f"{vp_addr} missing from ripe atlas set")

    if points:
        return weighted_centroid(points)

    return None, None
