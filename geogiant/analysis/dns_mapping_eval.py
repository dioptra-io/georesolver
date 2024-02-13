import asyncio
import numpy as np

from tqdm import tqdm
from pyasn import pyasn
from collections import defaultdict
from loguru import logger

from geogiant.analysis.utils import get_pings_per_target
from geogiant.analysis.plot import plot_median_error_per_finger_printing_method
from geogiant.common.geoloc import distance
from geogiant.common.settings import PathSettings, ClickhouseSettings
from geogiant.common.files_utils import load_json, load_pickle, load_csv, dump_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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


def ecs_dns_vp_selection_eval(
    targets: list,
    vps_subnet: dict,
    vps_bgp_prefix: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    vp_coordinates: dict,
    probing_budget: int = 20,
) -> tuple[list, set]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    unmapped_t = set()
    for target in targets:
        target_addr = target["address_v4"]
        target_subnet = get_prefix_from_ip(target_addr)
        asn, target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)

        # get target/vps dns mapping score
        try:
            target_scores = subnet_scores[target_bgp_prefix]
        except KeyError:
            unmapped_t.add(target_addr)
            continue

        # if np.mean([score for _, score in target_scores[:10]]) < 0.5:
        #     continue

        # retrieve all vps belonging to subnets with highest mapping scores
        vps_assigned = []
        for vp_bgp_prefix, _ in target_scores[:probing_budget]:
            vps_assigned.extend([vp_addr for vp_addr in vps_bgp_prefix[vp_bgp_prefix]])

        # logger.debug(target_scores[:probing_budget])

        # get vps and pings associated to the dns mapping selection
        try:
            vp_selection = get_ecs_pings(
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
        best_vp_addr, best_rtt = min(
            ping_vps_to_target[target_addr], key=lambda x: x[-1]
        )

        for index, vp in enumerate(ping_vps_to_target[target_addr]):
            min_rtt = vp[-1]
            if min_rtt <= 2:
                break

        threshold_vp_index = index
        threshold_vp = vp

        best_vp_subnet = get_prefix_from_ip(best_vp_addr)

        best_vp_score = -1
        best_vp_index = -1
        for i, (vp_subnet, score) in enumerate(target_scores):
            if best_vp_subnet == vp_subnet:
                best_vp_score = score
                best_vp_index = i
                break

        best_elected_vp_score = -1
        best_elected_vp_index = -1
        for i, (vp_subnet, score) in enumerate(target_scores):
            if get_prefix_from_ip(best_elected_vp[0]) == vp_subnet:
                best_elected_vp_score = score
                best_elected_vp_index = i
                break

        min_elected_rtt = best_elected_vp[-1]

        target_lon, target_lat = target["geometry"]["coordinates"]
        elect_vp_lat, elect_vp_lon = vp_coordinates[best_elected_vp[0]]

        try:
            best_vp_lat, best_vp_lon = vp_coordinates[best_vp_addr]
        except KeyError:
            continue

        d_elected_error = distance(target_lat, elect_vp_lat, target_lon, elect_vp_lon)
        best_d_error = distance(target_lat, best_vp_lat, target_lon, best_vp_lon)

        results[target_addr] = {
            "target_subnet": target_subnet,
            "target_bgp_prefix": target_bgp_prefix,
            "elected_vps": vp_selection,
            "best_elected_vp": best_elected_vp[0],
            "best_elected_vp_score": best_elected_vp_score,
            "best_elected_vp_index": best_elected_vp_index,
            "elected_min_rtt": min_elected_rtt,
            "elected_d_error": d_elected_error,
            "best_vp": best_vp_addr,
            "best_vp_score": best_vp_score,
            "best_vp_index": best_vp_index,
            "best_d_error": best_d_error,
            "threshold_vp": threshold_vp,
            "threshold_vp_index": threshold_vp_index,
        }

    return results, unmapped_t


def get_metrics_vp_selection(
    targets: list,
    vps_subnet: dict,
    vps_bgp_prefix: dict,
    ping_vps_to_target: dict,
    subnet_scores: dict,
    vp_coordinates: dict,
    probing_budgets: list,
    verbose: bool = False,
) -> dict:
    """calculate some basic metrics from a vp selection"""
    overall_results = {}
    hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames_cdn.csv")
    hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
    hostname_filter = f"AND hostname IN ({hostname_filter})"
    for budget in probing_budgets:
        if budget == 0:
            continue

        ecs_results, unmapped_t = ecs_dns_vp_selection_eval(
            targets=targets,
            vps_subnet=vps_subnet,
            vps_bgp_prefix=vps_bgp_prefix,
            subnet_scores=subnet_scores,
            ping_vps_to_target=ping_vps_to_target,
            vp_coordinates=vp_coordinates,
            probing_budget=budget,
        )

        median_distance = round(
            np.median([r["elected_d_error"] for r in ecs_results.values()]), 2
        )
        deviation = round(
            np.std([r["elected_d_error"] for r in ecs_results.values()]), 2
        )

        overall_results[budget] = ecs_results

        # Debugging
        if verbose:
            for target_addr, target_results in ecs_results.items():
                diff_error = (
                    target_results["elected_d_error"] - target_results["best_d_error"]
                )
                if diff_error > 100:
                    logger.info("######################################")
                    logger.info(f"Target : {target_addr}")
                    logger.info(
                        f"Target BGP prefix : {target_results['target_bgp_prefix']}"
                    )
                    logger.info("ECS-DNS elected VP:")
                    logger.info(f"addr    = {target_results['best_elected_vp']}")
                    logger.info(f"score   = {target_results['best_elected_vp_score']}")
                    logger.info(f"index   = {target_results['best_elected_vp_index']}")
                    logger.info(f"d error = {target_results['elected_d_error']}")
                    logger.info("Min RTT VP:")
                    logger.info(f"addr    = {target_results['best_vp']}")
                    logger.info(f"score   = {target_results['best_vp_score']}")
                    logger.info(f"index   = {target_results['best_vp_index']}")
                    logger.info(f"d error = {target_results['best_d_error']}")
                    logger.info(f"diff error dist: {diff_error}")
                    logger.info(
                        f"SELECT client_subnet, hostname, answer_subnet, answer_bgp_prefix, pop_ip_info_id, pop_city from {clickhouse_settings.DATABASE}.{'test_clustering'} where (client_subnet == toIPv4('{target_results['target_subnet']}') OR client_subnet == toIPv4('{get_prefix_from_ip(target_results['best_elected_vp'])}') OR client_subnet == toIPv4('{get_prefix_from_ip(target_results['best_vp'])}')) {hostname_filter} order by (hostname, client_subnet)"
                    )
                    logger.info("######################################")

        # debugging
        if budget in [10, 50, 100, 500, 1_000]:
            logger.info(f"probing budget={budget} Median dist={median_distance}")

    return overall_results


async def main(targets: list, vps: list) -> None:
    asndb = pyasn(str(path_settings.RIB_TABLE))
    probing_budgets = [1]
    probing_budgets.extend([i for i in range(10, 100, 10)])
    probing_budgets.extend([i for i in range(100, 500, 100)])

    # probing_budgets = [10]

    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    vps_bgp_prefix = defaultdict(list)
    for vp in vps:
        addr = vp["address_v4"]
        subnet = get_prefix_from_ip(addr)
        _, vp_bgp_prefix = route_view_bgp_prefix(addr, asndb)
        vp_lon, vp_lat = vp["geometry"]["coordinates"]

        if not vp_bgp_prefix:
            continue

        vps_subnet[subnet].append(addr)
        vps_bgp_prefix[vp_bgp_prefix].append(addr)
        vps_coordinates[addr] = (vp_lat, vp_lon)

    ping_vps_to_target = await get_pings_per_target(
        clickhouse_settings.OLD_PING_VPS_TO_TARGET
    )

    eval_results = {}
    unfiltered_eval_results = {}
    logger.info("Answers score evaluation")
    # eval_results["answers"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(
    #         path_settings.RESULTS_PATH / "overall_answers_score.pickle"
    #     ),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    # logger.info("Unfiltered Answers score evaluation")
    # unfiltered_eval_results["unfiltered_answers"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(
    #         path_settings.RESULTS_PATH / "unfiltered_overall_answers_score.pickle"
    #     ),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    ecs_dns_scores = load_pickle(path_settings.RESULTS_PATH / "ecs_dns_scores.pickle")

    logger.info("BGP score evaluation")
    eval_results["bgp_prefix"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        vps_bgp_prefix=vps_bgp_prefix,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=ecs_dns_scores["client_bgp_prefix"]["answer_bgp_prefix"],
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
        verbose=False,
    )

    logger.info("BGP score evaluation")
    eval_results["bgp_prefix"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        vps_bgp_prefix=vps_bgp_prefix,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=ecs_dns_scores["client_bgp_prefix"]["answer_bgp_prefix"],
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
        verbose=False,
    )

    # logger.info("Unfiltered Subnet score evaluation")
    # unfiltered_eval_results["unfiltered_subnet"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(
    #         path_settings.RESULTS_PATH / "unfiltered_overall_subnet_score.pickle"
    #     ),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    # logger.info("BGP prefix score evaluation")
    # eval_results["bgp_prefix"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     vps_bgp_prefix=vps_bgp_prefix,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(
    #         path_settings.RESULTS_PATH / "overall_bgp_prefix_score.pickle"
    #     ),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    #     verbose=False,
    # )

    # logger.info("Unfiltered BGP prefix score evaluation")
    # unfiltered_eval_results["unfiltered_bgp_prefix"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(
    #         path_settings.RESULTS_PATH / "unfiltered_overall_bgp_prefix_score.pickle"
    #     ),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    # eval_results["pop_id"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(path_settings.RESULTS_PATH / "pop_id_score.pickle"),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    dump_pickle(eval_results, path_settings.RESULTS_PATH / "ecs_dns_eval.pickle")


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(main(targets, vps))
