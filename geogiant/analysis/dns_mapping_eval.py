import asyncio
import numpy as np

from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.analysis.plot import plot_median_error_per_finger_printing_method
from geogiant.clickhouse import GetPingsPerTarget
from geogiant.common.geoloc import rtt_to_km, distance
from geogiant.common.settings import PathSettings, ClickhouseSettings
from geogiant.common.files_utils import load_json, load_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def get_pings_per_target_parsed(table_name: str) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetPingsPerTarget().execute(
            client=client,
            table_name=table_name,
        )

    for row in resp:
        ping_vps_to_target[row["target"]] = row["pings"]

    return ping_vps_to_target


def get_ping_to_target(
    target_associated_vps: list,
    ping_to_target: list,
) -> list:
    vp_selection = []

    # filter out all vps not included by ecs-dns methodology
    for vp_addr, min_rtt in ping_to_target:
        if vp_addr in target_associated_vps:
            vp_selection.append((vp_addr, min_rtt))

    return vp_selection


def ecs_dns_selection_evaluation(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    vp_coordinates: dict,
    probing_budget: int = 20,
) -> tuple[list, set]:
    results = []
    unmapped_t = set()
    for target in targets:
        target_addr = target["address_v4"]
        target_subnet = get_prefix_from_ip(target_addr)

        # get target/vps dns mapping score
        try:
            target_scores = subnet_scores[target_subnet]
        except KeyError:
            unmapped_t.add(target_addr)
            continue

        # retrieve all vps belonging to subnets with highest mapping scores
        vps_assigned = []
        for vp_subnet, _ in target_scores[:probing_budget]:
            vps_assigned.extend([vp_addr for vp_addr in vps_subnet[vp_subnet]])

        # get vps and pings associated to the dns mapping selection
        try:
            vp_selection = get_ping_to_target(
                target_associated_vps=vps_assigned,
                ping_to_target=ping_vps_to_target[target_addr],
            )
        except KeyError:
            unmapped_t.add(target_addr)
            continue

        # if no vps, save target as unmapped
        if not vp_selection:
            unmapped_t.add(target_addr)
            continue

        # get the best vp, min rtt and compute distance error
        best_elected_vp = min(vp_selection, key=lambda x: x[-1])
        best_vp_addr, best_min_rtt = min(
            ping_vps_to_target[target_addr], key=lambda x: x[-1]
        )
        best_vp_subnet = get_prefix_from_ip(best_vp_addr)
        best_d_error = rtt_to_km(best_min_rtt)

        best_vp_score = -1
        best_vp_index = -1
        for i, (vp_subnet, score) in enumerate(target_scores):
            if best_vp_subnet == vp_subnet:
                best_vp_score = score
                best_vp_index = i
                break

        min_elected_rtt = best_elected_vp[-1]
        rtt_to_dist_elected = rtt_to_km(min_elected_rtt)

        target_lon, target_lat = target["geometry"]["coordinates"]
        elect_vp_lat, elect_vp_lon = vp_coordinates[best_elected_vp[0]]

        d_elected_error = distance(target_lat, elect_vp_lat, target_lon, elect_vp_lon)

        results.append(
            {
                "target": target_addr,
                "elected_vps": vp_selection,
                "d_error_elected": d_elected_error,
                "rtt_to_dist_elected": rtt_to_dist_elected,
                "best_vp_addr": best_vp_addr,
                "best_vp_score": best_vp_score,
                "best_vp_index": best_vp_index,
                "best_lat_to_dist": best_d_error,
            }
        )

    return results, unmapped_t


def get_metrics_vp_selection(
    targets: list,
    vps_subnet: dict,
    ping_vps_to_target: dict,
    subnet_scores: dict,
    vp_coordinates: dict,
    probing_budgets: list,
) -> dict:
    """calculate some basic metrics from a vp selection"""
    overall_results = {}
    for budget in probing_budgets:
        if budget == 0:
            continue

        ecs_results, unmapped_t = ecs_dns_selection_evaluation(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores,
            ping_vps_to_target=ping_vps_to_target,
            vp_coordinates=vp_coordinates,
            probing_budget=budget,
        )

        median_distance = round(
            np.median([r["d_error_elected"] for r in ecs_results]), 2
        )
        deviation = round(np.std([r["d_error_elected"] for r in ecs_results]), 2)

        overall_results[budget] = (median_distance, deviation)

        # debugging
        if budget in [10, 50, 100, 500, 1_000]:
            logger.info("ECS DNS vp selection per target with target pop signature")
            logger.info(
                f"nb targets: {len(targets)} | unmapped targets: {len(unmapped_t)}"
            )
            logger.info(
                f"Median min dist per targets: {median_distance} | probing budget: {budget} \n"
            )

    return overall_results


async def main(targets: list, vps: list) -> None:
    probing_budgets = [i for i in range(0, 100, 10)]
    probing_budgets.extend([i for i in range(100, 1_000, 50)])

    vps_subnet = defaultdict(list)
    vps_coordinates = defaultdict(list)
    for vp in vps:
        addr = vp["address_v4"]
        subnet = get_prefix_from_ip(addr)
        vp_lon, vp_lat = vp["geometry"]["coordinates"]

        vps_subnet[subnet].append(addr)
        vps_coordinates[addr] = (vp_lat, vp_lon)

    ping_vps_to_target = await get_pings_per_target_parsed(
        clickhouse_settings.OLD_PING_VPS_TO_TARGET
    )

    eval_results = {}
    eval_results["answers"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "answers_score.pickle"),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    eval_results["subnet"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "subnet_score.pickle"),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    eval_results["bgp_prefix"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(
            path_settings.RESULTS_PATH / "bgp_prefix_score.pickle"
        ),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    eval_results["pop_id"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "pop_id_score.pickle"),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    plot_median_error_per_finger_printing_method(eval_results)

    # answers_score = load_pickle(path_settings.RESULTS_PATH / "answers_score.pickle")
    # subnet_score = load_pickle(path_settings.RESULTS_PATH / "subnet_score.pickle")
    # bgp_prefix_score = load_pickle(
    #     path_settings.RESULTS_PATH / "bgp_prefix_score.pickle"
    # )
    # pop_id_score = load_pickle(path_settings.RESULTS_PATH / "pop_id_score.pickle")


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(main(targets, vps))
