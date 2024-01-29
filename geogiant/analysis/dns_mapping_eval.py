import asyncio
import numpy as np

from pyasn import pyasn
from collections import defaultdict
from loguru import logger

from geogiant.analysis.utils import get_pings_per_target
from geogiant.analysis.plot import plot_median_error_per_finger_printing_method
from geogiant.common.geoloc import rtt_to_km, distance
from geogiant.common.settings import PathSettings, ClickhouseSettings
from geogiant.common.files_utils import load_json, load_pickle
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
        target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)

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

        for i, (vp_subnet, score) in enumerate(target_scores):
            if get_prefix_from_ip(best_elected_vp[0]) == vp_subnet:
                best_elected_vp_score = score
                best_elected_vp_index = i
                break

        min_elected_rtt = best_elected_vp[-1]
        rtt_to_dist_elected = rtt_to_km(min_elected_rtt)

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
        }

    return results, unmapped_t


def get_metrics_vp_selection(
    targets: list,
    vps_subnet: dict,
    ping_vps_to_target: dict,
    subnet_scores: dict,
    vp_coordinates: dict,
    probing_budgets: list,
    verbose: bool = False,
) -> dict:
    """calculate some basic metrics from a vp selection"""
    overall_results = {}
    hostname_filter = (
        "outlook.live.com",
        "docs.edgecast.com",
        "www.office.com",
        "advancedhosting.com",
        "tencentcloud.com",
        "teams.microsoft.com",
        "news.yahoo.co.jp",
        "baseball.yahoo.co.jp",
        "www.yahoo.co.jp",
        "finance.yahoo.co.jp",
        "weather.yahoo.co.jp",
        "cachefly.com",
        "fastly.com",
        "forms.office.com",
        "detail.chiebukuro.yahoo.co.jp",
        "sports.yahoo.co.jp",
        "weather.yahoo.co.jp",
        "auctions.yahoo.co.jp",
        "page.auctions.yahoo.co.jp",
        "search.yahoo.co.jp",
        "www.facebook.com",
        "www.instagram.com",
        "www.google.pl",
        "www.google.ca",
        "www.google.cl",
        "www.google.co.in",
        "www.google.co.jp",
        "www.google.com.ar",
        "www.google.co.uk",
        "www.google.com.tw",
        "www.google.de",
        "www.google.fr",
        "www.google.it",
        "www.google.nl",
        "www.google.p",
        "www.google.com.br",
        "www.google.es",
        "www.google.com.mx",
        "www.google.com.hk",
        "www.google.co.th",
        "www.google.com.tr",
    )
    hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
    hostname_filter = f"AND hostname NOT IN ({hostname_filter})"
    for budget in probing_budgets:
        if budget == 0:
            continue

        ecs_results, unmapped_t = ecs_dns_vp_selection_eval(
            targets=targets,
            vps_subnet=vps_subnet,
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

        overall_results[budget] = (median_distance, deviation)

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
                        f"SELECT client_subnet, hostname, answer_subnet, answer_bgp_prefix, pop_ip_info_id from {clickhouse_settings.DATABASE}.{clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA} where (client_subnet == toIPv4('{target_results['target_subnet']}') OR client_subnet == toIPv4('{get_prefix_from_ip(target_results['best_elected_vp'])}') OR client_subnet == toIPv4('{get_prefix_from_ip(target_results['best_vp'])}')) {hostname_filter} order by (hostname, client_subnet)"
                    )
                    logger.info("######################################")

        # debugging
        if budget in [10, 50, 100, 500, 1_000]:
            logger.info("ECS DNS vp selection per target with target pop signature")
            logger.info(f"probing budget={budget} Median dist={median_distance}")

    return overall_results


async def main(targets: list, vps: list) -> None:
    probing_budgets = [i for i in range(0, 100, 10)]
    probing_budgets.extend([i for i in range(100, 500, 50)])
    # probing_budgets = [50]

    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    for vp in vps:
        addr = vp["address_v4"]
        subnet = get_prefix_from_ip(addr)
        vp_lon, vp_lat = vp["geometry"]["coordinates"]

        vps_subnet[subnet].append(addr)
        vps_coordinates[addr] = (vp_lat, vp_lon)

    ping_vps_to_target = await get_pings_per_target(
        clickhouse_settings.OLD_PING_VPS_TO_TARGET
    )

    eval_results = {}
    logger.info("Answers score evaluation")
    eval_results["answers"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "answers_score.pickle"),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    logger.info("Unfiltered Answers score evaluation")
    eval_results["unfiltered_answers"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(
            path_settings.RESULTS_PATH / "unfiltered_answers_score.pickle"
        ),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    logger.info("Subnet score evaluation")
    eval_results["subnet"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(path_settings.RESULTS_PATH / "subnet_score.pickle"),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    logger.info("Unfiltered Subnet score evaluation")
    eval_results["unfiltered_subnet"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(
            path_settings.RESULTS_PATH / "unfiltered_subnet_score.pickle"
        ),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    logger.info("BGP prefix score evaluation")
    eval_results["bgp_prefix"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(
            path_settings.RESULTS_PATH / "bgp_prefix_score.pickle"
        ),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
        verbose=False,
    )

    logger.info("Unfiltered BGP prefix score evaluation")
    eval_results["unfiltered_bgp_prefix"] = get_metrics_vp_selection(
        targets=targets,
        vps_subnet=vps_subnet,
        ping_vps_to_target=ping_vps_to_target,
        subnet_scores=load_pickle(
            path_settings.RESULTS_PATH / "unfiltered_bgp_prefix_score.pickle"
        ),
        vp_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    # eval_results["pop_id"] = get_metrics_vp_selection(
    #     targets=targets,
    #     vps_subnet=vps_subnet,
    #     ping_vps_to_target=ping_vps_to_target,
    #     subnet_scores=load_pickle(path_settings.RESULTS_PATH / "pop_id_score.pickle"),
    #     vp_coordinates=vps_coordinates,
    #     probing_budgets=probing_budgets,
    # )

    plot_median_error_per_finger_printing_method(
        eval_results, out_file="mapping_scores_evaluation.pdf"
    )


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(main(targets, vps))
