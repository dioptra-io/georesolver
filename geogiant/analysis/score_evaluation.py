import asyncio
import numpy as np

from dataclasses import dataclass
from pyasn import pyasn
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

from geogiant.ecs_vp_selection.query import (
    get_pings_per_target,
    load_targets,
    load_vps,
)
from geogiant.ecs_vp_selection.utils import (
    get_ecs_pings,
    select_one_vp_per_as_city,
    get_parsed_vps,
    ResultsScore,
)
from geogiant.common.geoloc import distance
from geogiant.common.files_utils import load_pickle, dump_pickle, load_json
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import PathSettings, ClickhouseSettings

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


def get_ecs_vps(
    target_score: dict, vps_per_subnet: dict, probing_budget: int = 50
) -> list:
    """
    get the target score and extract best VPs function of the probing budget
    return 1 VP per subnet, TODO: get best connected VP per subnet
    """
    # retrieve all vps belonging to subnets with highest mapping scores
    ecs_vps = []
    for subnet, score in target_score[:probing_budget]:
        vps_in_subnet = vps_per_subnet[subnet]
        if not vps_in_subnet:
            continue
        ecs_vps.append((vps_in_subnet[0], score))

    return ecs_vps


def shortest_ping(selected_vps: list, pings: dict) -> tuple:
    """return the shortest ping for a selection of VPs and a set of measurements"""
    try:
        selected_pings = get_ecs_pings(
            target_associated_vps=selected_vps,
            ping_to_target=pings,
        )
    except KeyError:
        return None

    if not selected_pings:
        return None, None

    addr, min_rtt = min(selected_pings, key=lambda x: x[-1])

    return addr, min_rtt


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
    lat, lon, _ = vps_coordinates[vp_addr]
    d_error = distance(target["lat"], lat, target["lon"], lon)
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


def get_no_ping_vp(
    target, target_score: list, vps_per_subnet: dict, vps_coordinates: dict
) -> dict:
    """return VP with maximum score"""
    subnet, _ = target_score[0]
    vp_addr = vps_per_subnet[subnet][0]

    return get_vp_info(target, target_score, vp_addr, vps_coordinates)


def parse_target(target: dict, asndb: pyasn) -> dict:
    """simply get target into a nice dict structure"""
    addr = target["address_v4"]
    subnet = get_prefix_from_ip(addr)
    bgp_prefix = route_view_bgp_prefix(subnet, asndb)

    return {
        "addr": addr,
        "subnet": subnet,
        "bgp_prefix": bgp_prefix,
        "lat": target["lat"],
        "lon": target["lon"],
    }


def ecs_dns_vp_selection_eval(
    targets: list,
    vps_per_subnet: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    vps_coordinates: dict,
    probing_budgets: list,
) -> tuple[dict, dict]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    for target in tqdm(targets):

        target = parse_target(target, asndb)

        try:
            target_score = subnet_scores[target["subnet"]]
            target_score = target_score["intersection"]
        except KeyError:
            logger.error(f"cannot find target score for subnet : {target['subnet']}")
            continue

        ecs_vps = get_ecs_vps(target_score, vps_per_subnet, 1_00)

        filtered_ecs_vps = []
        for vp_subnet, score in ecs_vps:
            if vp_subnet != target["subnet"]:
                filtered_ecs_vps.append((vp_subnet, score))

        ecs_vps = filtered_ecs_vps

        # Once cluster done with 50 VPs, select VPs function of probing budget
        ecs_vps_per_budget = {}
        for budget in probing_budgets:
            ecs_vps_per_budget[budget] = ecs_vps[:budget]
        # Best score + AS + city
        ecs_vps_budget = select_one_vp_per_as_city(ecs_vps_budget, vps_coordinates)

        # SHORTEST PING SELECTION
        try:
            ref_shortest_ping_addr, ref_min_rtt = shortest_ping(
                list(vps_coordinates.keys()), ping_vps_to_target[target["addr"]]
            )

            ecs_shortest_ping_per_budget = {}
            for budget, ecs_vps in ecs_vps_per_budget.items():
                ecs_shortest_ping_per_budget[budget] = shortest_ping(
                    [addr for addr, _ in ecs_vps_budget],
                    ping_vps_to_target[target["addr"]],
                )

        except KeyError:
            logger.debug(f"No ping available for target:: {target['addr']}")
            continue

        if not ref_shortest_ping_addr:
            logger.debug(f"no ping retrieved for target:: {target['addr']}")
            continue

        ref_shortest_ping_vp = get_vp_info(
            target,
            target_score,
            ref_shortest_ping_addr,
            vps_coordinates,
            ref_min_rtt,
        )

        ecs_shortest_ping_per_budget = {}
        for budget, (
            ecs_shortest_ping_addr,
            ecs_min_rtt,
        ) in ecs_shortest_ping_per_budget.items():
            ecs_shortest_ping_vp = get_vp_info(
                target,
                target_score,
                ecs_shortest_ping_addr,
                vps_coordinates,
                ecs_min_rtt,
            )
            ecs_shortest_ping_per_budget[budget] = ecs_shortest_ping_vp

        no_ping_vp = get_no_ping_vp(
            target,
            target_score,
            vps_per_subnet,
            vps_coordinates,
        )

        results[target["addr"]] = {
            "target": target,
            "ref_shortest_ping_vp": ref_shortest_ping_vp,
            "ecs_shortest_ping_vp_per_budget": ecs_shortest_ping_per_budget,
            "no_ping_vp": no_ping_vp,
            "ecs_scores": target_score[:50],
            "ecs_vps": ecs_vps,
            "ecs_vps_budget": ecs_vps_budget,
        }

    return results


async def main() -> None:
    probing_budgets = [50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    ping_vps_to_target = await get_pings_per_target(
        clickhouse_settings.OLD_PING_VPS_TO_TARGET
    )
    targets = await load_targets(clickhouse_settings.VPS_RAW)
    vps = await load_vps(clickhouse_settings.VPS_RAW)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)

    eval_results = {}
    logger.info("BGP prefix score geoloc evaluation")

    score_files = [
        "scores_AMAZON-02_1_greedy_per_cdn.pickle",
        "scores_AMAZON-02_5_greedy_per_cdn.pickle",
        "scores_AMAZON-02_10_greedy_per_cdn.pickle",
    ]

    for score_file in score_files:

        scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)

        results = ResultsScore()

        if "answers" in scores.score_config["answer_granularities"]:
            results_answers = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        if "answer_subnets" in scores.score_config["answer_granularities"]:
            results_answer_subnets = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        if "answer_bgp_prefixes" in scores.score_config["answer_granularities"]:
            results_answer_bgp_prefixes = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        results = EvalResults(
            target_scores=scores,
            results_answers=results_answers,
            results_answer_subnets=results_answer_subnets,
            results_answer_bgp_prefixes=results_answer_bgp_prefixes,
        )

        dump_pickle(
            data=results,
            output_file=path_settings.RESULTS_PATH
            / f"{'results_' + score_file.split('score')[-1]}.pickle",
        )


if __name__ == "__main__":
    asyncio.run(main())
