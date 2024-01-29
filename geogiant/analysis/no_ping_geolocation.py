import asyncio
import numpy as np

from collections import defaultdict
from loguru import logger

from geogiant.analysis.utils import get_pings_per_target
from geogiant.analysis.plot import plot_median_error_per_finger_printing_method
from geogiant.common.geoloc import (
    rtt_to_km,
    distance,
    polygon_centroid,
    weighted_centroid,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings
from geogiant.common.files_utils import load_json, load_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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

    target_lat, target_lon = polygon_centroid(points)

    return (target_lat, target_lon)


def get_vp_weight(
    vp_i: str, vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> float:
    """
    get vp weight for centroid calculation.
    we take w = 1 / sum(d_i_j)
    """
    vp_i_lat, vp_i_lon = vps_coordinate[vp_i]
    sum_d = 0
    for vp_j in vps_assigned:
        if vp_j == vp_i:
            continue

        try:
            vp_j_lat, vp_j_lon = vps_coordinate[vp_j]
        except KeyError:
            continue

        sum_d += distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

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

    target_lat, target_lon = weighted_centroid(points)
    return (target_lat, target_lon)


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


def no_pings_eval(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    vps_coordinates: dict,
    probing_budget: int = 20,
) -> tuple[dict, dict]:
    results = {}
    w_results = {}
    b_results = {}
    for target in targets:
        target_addr = target["address_v4"]
        target_subnet = get_prefix_from_ip(target_addr)
        target_lon, target_lat = target["geometry"]["coordinates"]
        target_scores = subnet_scores[target_subnet]

        # retrieve all vps belonging to subnets with highest mapping scores
        vps_assigned = []
        for vp_subnet, _ in target_scores[:probing_budget]:
            vps_assigned.extend([vp_addr for vp_addr in vps_subnet[vp_subnet]])

        # estimate target geoloc based on ecs score
        ecs_lat, ecs_lon = ecs_target_geoloc(vps_assigned, vps_coordinates)
        w_ecs_lat, w_ecs_lon = weighted_ecs_target_geoloc(vps_assigned, vps_coordinates)
        b_ecs_lat, b_ecs_lon = best_ecs_target_geoloc(vps_assigned, vps_coordinates)

        # compare true target geolocation with estimated one
        d_error = distance(target_lat, ecs_lat, target_lon, ecs_lon)
        w_d_error = distance(target_lat, w_ecs_lat, target_lon, w_ecs_lon)
        b_d_error = distance(target_lat, b_ecs_lat, target_lon, b_ecs_lon)

        results[target_addr] = d_error
        w_results[target_addr] = w_d_error
        b_results[target_addr] = b_d_error

    return results, w_results, b_results


def get_metrics(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    vps_coordinates: dict,
    probing_budgets: list,
) -> dict:
    """get geolocation error function of the probing budget"""
    overall_results = {}
    for budget in probing_budgets:
        if budget == 0:
            continue

        no_ping_r, w_no_ping_r, b_no_ping_r = no_pings_eval(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores,
            vps_coordinates=vps_coordinates,
            probing_budget=budget,
        )

        median_d = round(np.median([d_error for d_error in no_ping_r.values()]), 2)
        w_median_d = round(np.median([d_error for d_error in w_no_ping_r.values()]), 2)
        b_median_d = round(np.median([d_error for d_error in b_no_ping_r.values()]), 2)

        overall_results[budget] = (median_d, w_median_d)

        # debugging
        if budget in [5, 10, 20, 30, 50, 100]:
            logger.info(f"probing budget: {budget}")
            logger.info(f"Median error: {median_d}")
            logger.info(f"Weight Median error: {w_median_d}")
            logger.info(f"Highest score vp Median error: {b_median_d}")

    return overall_results


async def main(targets: list, vps: list) -> None:
    probing_budgets = [i for i in range(5, 50, 1)]
    probing_budgets.extend([i for i in range(50, 100, 10)])

    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    for vp in vps:
        addr = vp["address_v4"]
        subnet = get_prefix_from_ip(addr)
        vp_lon, vp_lat = vp["geometry"]["coordinates"]
        vps_coordinates[addr] = (vp_lat, vp_lon)
        vps_subnet[subnet].append(addr)

    eval_results = {}

    logger.info("Answers score geoloc evaluation")
    eval_results["answers"] = get_metrics(
        targets=targets,
        vps_subnet=vps_subnet,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "answers_score.pickle"),
        vps_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    logger.info("Subnet score geoloc evaluation")
    eval_results["subnet"] = get_metrics(
        targets=targets,
        vps_subnet=vps_subnet,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "subnet_score.pickle"),
        vps_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    logger.info("BGP prefix score geoloc evaluation")
    eval_results["bgp_prefix"] = get_metrics(
        targets=targets,
        vps_subnet=vps_subnet,
        subnet_scores=load_pickle(
            path_settings.RESULTS_PATH / "bgp_prefix_score.pickle"
        ),
        vps_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    # logger.info("POP id score geoloc evaluation")
    # eval_results["pop_id"] = get_metrics(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     subnet_scores=load_pickle(path_settings.RESULTS_PATH / "pop_id_score.pickle"),
    #     vps_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    plot_median_error_per_finger_printing_method(
        eval_results, out_file="no_pings_evaluation.pdf"
    )


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(main(targets, vps))
