from pyasn import pyasn
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger

from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.geoloc import distance


@dataclass(frozen=True)
class TargetScores:
    """
    dataclass object for storing a given score methodology.
    Score depends on the set of selected hostnames.
    """

    score_config: dict
    hostnames: list[str]
    cdns: list[str]
    score_answers: dict[list]
    score_answer_subnets: dict[list]
    score_answer_bgp_prefixes: dict[list]


@dataclass(frozen=True)
class EvalResults:
    """
    dataclass object for storing a given results methodology.
    Results depends on score calculation.
    """

    target_scores: TargetScores
    results_answers: list[dict]
    results_answer_subnets: list[dict]
    results_answer_bgp_prefixes: list[dict]


def parse_target(target: dict, asndb: pyasn) -> dict:
    """simply get target into a nice dict structure"""
    addr = target["addr"]
    subnet = get_prefix_from_ip(addr)
    bgp_prefix = route_view_bgp_prefix(subnet, asndb)

    return {
        "addr": addr,
        "subnet": subnet,
        "bgp_prefix": bgp_prefix,
        "lat": target["lat"],
        "lon": target["lon"],
    }


def get_parsed_vps(vps: list, asndb: pyasn, removed_vps: list = None) -> dict:
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

        vps_subnet[subnet].append(vp_addr)
        vps_bgp_prefix[vp_bgp_prefix].append(vp_addr)
        vps_coordinates[vp_addr] = (vp_lat, vp_lon, vp_asn)

    return vps_subnet, vps_coordinates


def get_vp_score(vp_subnet: str, target_score: list) -> tuple[float, float]:
    """retrieve vp score"""
    score = -1
    index = -1
    for i, (subnet, score) in enumerate(target_score):
        if vp_subnet == subnet:
            score = score
            index = i
            break

    return score, index


def get_vp_info(
    target: dict,
    target_score: list,
    vp_addr: str,
    vps_coordinates: dict,
    rtt: float = None,
) -> dict:
    """get all useful information about a selected VP"""
    vp_subnet = get_prefix_from_ip(vp_addr)
    try:
        lat, lon, _ = vps_coordinates[vp_addr]
    except KeyError:
        logger.debug(f"{vp_addr} present into pings but not in original VPs")
        return None

    d_error = distance(target["lat"], lat, target["lon"], lon)

    score, index = -1, -1
    if target_score:
        score, index = get_vp_score(vp_subnet, target_score)

    return {
        "addr": vp_addr,
        "subnet": vp_subnet,
        "lat": lat,
        "lon": lon,
        "rtt": rtt,
        "d_error": d_error,
        "score": score,
        "index": index,
    }


def filter_vps_last_mile_delay(
    ecs_vps: list[tuple], last_mile_delay: dict, rtt_thresholdd: int = 4
) -> list[tuple]:
    """remove vps that have a high last mile delay"""
    filtered_vps = []
    for vp_addr, score in ecs_vps:
        try:
            min_rtt = last_mile_delay[vp_addr]
            if min_rtt < rtt_thresholdd:
                filtered_vps.append((vp_addr, score))
        except KeyError:
            continue

    return filtered_vps


def select_one_vp_per_as_city(
    raw_vp_selection: list,
    vp_coordinates: dict,
    last_mile_delay: dict,
    threshold: int = 40,
) -> list:
    """from a list of VP, select one per AS and per city"""
    filtered_vp_selection = []
    vps_per_as = defaultdict(list)
    for vp_addr, score in raw_vp_selection:
        _, _, vp_asn = vp_coordinates[vp_addr]
        try:
            last_mile_delay_vp = last_mile_delay[vp_addr]
        except KeyError:
            continue

        vps_per_as[vp_asn].append((vp_addr, last_mile_delay_vp, score))

    # select one VP per AS, take maximum VP score in AS
    selected_vps_per_as = defaultdict(list)
    for asn, vps in vps_per_as.items():
        vps_per_as[asn] = sorted(vps, key=lambda x: x[1])
        for vp_i, last_mile_delay, score in vps_per_as[asn]:
            vp_i_lat, vp_i_lon, _ = vp_coordinates[vp_i]

            if not selected_vps_per_as[asn]:
                selected_vps_per_as[asn].append((vp_i, score))
                filtered_vp_selection.append((vp_i, score))
            else:
                already_found = False
                for vp_j, score in selected_vps_per_as[asn]:

                    vp_j_lat, vp_j_lon, _ = vp_coordinates[vp_j]

                    d = distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

                    if d < threshold:
                        already_found = True
                        break

                if not already_found:
                    selected_vps_per_as[asn].append((vp_i, score))
                    filtered_vp_selection.append((vp_i, score))

    return filtered_vp_selection


def get_vps_pings(
    target_associated_vps: list,
    ping_to_target: list,
) -> list:
    vp_selection = []

    # filter out all vps not included by ecs-dns methodology
    for vp_addr, min_rtt in ping_to_target:
        if vp_addr in target_associated_vps:
            vp_selection.append((vp_addr, min_rtt))

    return vp_selection


def shortest_ping(selected_vps: list, pings: dict) -> tuple:
    """return the shortest ping for a selection of VPs and a set of measurements"""
    try:
        selected_pings = get_vps_pings(
            target_associated_vps=selected_vps,
            ping_to_target=pings,
        )
    except KeyError:
        return None

    if not selected_pings:
        return None, None

    addr, min_rtt = min(selected_pings, key=lambda x: x[-1])

    return addr, min_rtt
