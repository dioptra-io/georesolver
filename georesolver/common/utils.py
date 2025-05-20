import asyncio

from random import sample
from pyasn import pyasn
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm
from numpy import mean

from georesolver.prober import RIPEAtlasAPI
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.clickhouse.queries import (
    get_pings_per_target,
    insert_pings,
    insert_traceroutes,
)
from georesolver.common.geoloc import distance
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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
    country_code = target["country_code"]
    subnet = get_prefix_from_ip(addr)
    bgp_prefix = route_view_bgp_prefix(subnet, asndb)

    return {
        "addr": addr,
        "subnet": subnet,
        "bgp_prefix": bgp_prefix,
        "country_code": country_code,
        "lat": target["lat"],
        "lon": target["lon"],
    }


def get_vps_country(vps: list) -> dict[str]:
    vps_country = {}
    for vp in vps:
        vps_country[vp["addr"]] = vp["country_code"]

    return vps_country


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


def get_vps_per_addr(vps: list) -> dict:
    """parse vps list to a dict for fast retrieval. Keys depends on granularity"""
    vps_coordinates = {}
    for vp in vps:
        vps_coordinates[vp["addr"]] = vp

    return vps_coordinates


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


def get_random_shortest_ping(
    targets: list[str], ping_table: str, removed_vps: list[str] = []
) -> None:
    """for each IP addresses retrieved the shortest ping"""
    ping_vps_to_target = get_pings_per_target(ping_table, removed_vps)

    shortest_ping_per_target = []
    for target_addr in tqdm(targets):
        try:
            target_pings = ping_vps_to_target[target_addr]
        except KeyError:
            continue

        target_pings = sample(
            target_pings, 50 if len(target_pings) >= 50 else len(target_pings)
        )

        _, shortest_ping_rtt = min(target_pings, key=lambda x: x[-1])
        shortest_ping_per_target.append((target_addr, shortest_ping_rtt))

    return shortest_ping_per_target


def get_shortest_ping_all_vp(
    targets: list[str], ping_table: str, removed_vps: list[str] = []
) -> None:
    """for each IP addresses retrieved the shortest ping"""
    ping_vps_to_target = get_pings_per_target(ping_table, removed_vps)

    shortest_ping_per_target = []
    for target_addr in tqdm(targets):
        try:
            target_pings = ping_vps_to_target[target_addr]
        except KeyError:
            continue

        _, shortest_ping_rtt = min(target_pings, key=lambda x: x[-1])
        shortest_ping_per_target.append((target_addr, shortest_ping_rtt))

    return shortest_ping_per_target


def get_vp_info(
    target: dict,
    target_score: list,
    vp_addr: str,
    vps_coordinates: dict,
    rtt: float = None,
    major_country: str = None,
) -> dict:
    """get all useful information about a selected VP"""
    vp_subnet = get_prefix_from_ip(vp_addr)
    try:
        lat, lon, _, _ = vps_coordinates[vp_addr]
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
        "country": target["country_code"],
        "major_country": major_country,
        "rtt": rtt,
        "d_error": d_error,
        "score": score,
        "index": index,
    }


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

    selected_pings = get_vps_pings(
        target_associated_vps=selected_vps,
        ping_to_target=pings,
    )

    if not selected_pings:
        return None, None

    addr, min_rtt = min(selected_pings, key=lambda x: x[-1])

    return addr, min_rtt


def get_ecs_vps(
    target_subnet: str,
    target_score: dict,
    vps_per_subnet: dict,
    last_mile_delay_vp: dict,
    probing_budget: int = 50,
) -> list:
    """
    get the target score and extract best VPs function of the probing budget
    """
    # retrieve all vps belonging to subnets with highest mapping scores
    ecs_vps = []
    # target_score = sorted(target_score, key=lambda x: x[1], reverse=True)
    for subnet, score in target_score:
        # for fairness, do not take vps that are in the same subnet as the target
        if subnet == target_subnet:
            continue

        vps_in_subnet = vps_per_subnet[subnet]

        vps_delay_subnet = []
        for vp in vps_in_subnet:
            try:
                vps_delay_subnet.append((vp, last_mile_delay_vp[vp]))
            except KeyError:
                continue

        # for each subnet, elect the VP with the lowest last mile delay
        if vps_delay_subnet:
            elected_subnet_vp_addr, _ = min(vps_delay_subnet, key=lambda x: x[-1])
            ecs_vps.append((elected_subnet_vp_addr, score))

        # take only a number of subnets up to probing budget
        if len(ecs_vps) >= probing_budget:
            break

    return ecs_vps


def get_d_errors_ref(
    pings_per_target: dict[list],
    vps_coordinates: dict[dict],
) -> list[float]:
    d_errors = []
    for target_addr, pings in pings_per_target.items():
        # get shortest ping using all vps
        vp_addr, _, _ = min(pings, key=lambda x: x[-1])

        # get target/vps coordinates
        try:
            target = vps_coordinates[target_addr]
            vp = vps_coordinates[vp_addr]
        except KeyError:
            continue

        # get dst
        d_error = distance(target["lat"], vp["lat"], target["lon"], vp["lon"])
        d_errors.append(d_error)

    return d_errors


def get_d_errors_georesolver(
    targets: list[str],
    pings_per_target: dict[list],
    vp_selection_per_target: dict[list],
    vps_coordinates: dict[dict],
    probing_budget: int = 50,
) -> tuple[list[float], list[float]]:
    """return distance errors based on a set of targets with known geolocation"""
    d_errors = []
    for target_addr in tqdm(targets):

        try:
            pings = pings_per_target[target_addr]
        except KeyError:
            logger.error(f"Cannot find pings for {target_addr=}")
            continue

        # geoResolver shortest ping
        try:
            if type(probing_budget) == int:
                vp_selection = vp_selection_per_target[target_addr][:probing_budget]
            elif type(probing_budget) == tuple:
                vp_selection = vp_selection_per_target[target_addr][
                    probing_budget[0] : probing_budget[1]
                ]
            else:
                raise RuntimeError(
                    f"Unsupported type {type(probing_budget)} for parameter probing_budget"
                )

        except KeyError:
            continue

        vp_selection = [v for v, _ in vp_selection]

        # get georesolver shortest ping
        georesolver_pings = []
        for vp_addr, vp_id, min_rtt in pings:
            if vp_addr in vp_selection:
                georesolver_pings.append((vp_addr, vp_id, min_rtt))
        try:
            vp_addr, _, _ = min(georesolver_pings, key=lambda x: x[-1])
        except:
            continue

        # get target/vps coordinates
        target = vps_coordinates[target_addr]
        vp = vps_coordinates[vp_addr]

        # get dst
        d_error = distance(target["lat"], vp["lat"], target["lon"], vp["lon"])
        d_errors.append(d_error)

    return d_errors
