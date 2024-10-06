import numpy as np
from collections import defaultdict
from loguru import logger
from pyasn import pyasn

from geogiant.common.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
    load_targets,
    load_vps,
)
from geogiant.common.utils import (
    get_parsed_vps,
    get_vps_country,
    EvalResults,
    TargetScores,
)
from geogiant.common.geoloc import distance
from geogiant.evaluation.plot import ecdf, plot_cdf, get_proportion_under
from geogiant.ecs_geoloc_eval import ecs_dns_vp_selection_eval
from geogiant.evaluation.scores import get_scores
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.files_utils import load_csv, load_json, load_pickle, dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings


path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    targets_table = clickhouse_settings.VPS_FILTERED_TABLE
    vps_table = clickhouse_settings.VPS_FILTERED_TABLE

    targets_ecs_table = "vps_mapping_ecs"
    vps_ecs_table = "vps_mapping_ecs"

    selected_hostnames_per_cdn_per_ns = load_json(
        path_settings.DATASET
        / f"hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json"
    )

    selected_hostnames = set()
    selected_hostnames_per_cdn = defaultdict(list)
    for ns in selected_hostnames_per_cdn_per_ns:
        for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
            selected_hostnames.update(hostnames)
            selected_hostnames_per_cdn[org].extend(hostnames)

    logger.info(f"{len(selected_hostnames)=}")

    output_path = (
        path_settings.RESULTS_PATH
        / f"tier5_evaluation/scores__best_hostname_geo_score.pickle"
    )

    score_config = {
        "targets_table": targets_table,
        "main_org_threshold": 0.0,
        "bgp_prefixes_threshold": 0.0,
        "vps_table": vps_table,
        "hostname_per_cdn_per_ns": selected_hostnames_per_cdn_per_ns,
        "hostname_per_cdn": selected_hostnames_per_cdn,
        "targets_ecs_table": targets_ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": ["jaccard"],
        "answer_granularities": ["answer_subnets"],
        "output_path": output_path,
    }

    get_scores(score_config)


def get_diff_error_per_budget() -> tuple[dict, dict]:

    eval: EvalResults = load_pickle(
        path_settings.RESULTS_PATH
        / "tier5_evaluation/results__best_hostname_geo_score.pickle"
    )

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_ref_shortest_ping.pickle"
    )

    d_errors_per_budget = defaultdict(list)
    target_diff_per_budget = defaultdict(list)
    ref_shortest_ping_indexes = []
    for target_addr, results_per_metric in eval.results_answer_subnets.items():

        # get ref info
        try:
            ref_shortest_ping_vp = ref_shortest_ping_results[target_addr][
                "ref_shortest_ping_vp"
            ]
        except KeyError:
            continue

        ref_shortest_ping_indexes.append(ref_shortest_ping_vp["index"])

        # only consider ip addresses above 40 km in ref
        if ref_shortest_ping_vp["d_error"] > 40:
            continue

        # get target results
        for _, target_results in results_per_metric["result_per_metric"].items():
            shortest_ping_vp_per_budget = target_results[
                "ecs_shortest_ping_vp_per_budget"
            ]

            for budget, shortest_ping_vp in shortest_ping_vp_per_budget.items():
                # only consider IP addresses that we do not geolocate under 40
                if shortest_ping_vp["d_error"] > 40:
                    error_diff = (
                        shortest_ping_vp["d_error"] - ref_shortest_ping_vp["d_error"]
                    )
                    nb_vps = len(target_results["ecs_vps"])
                    d_errors_per_budget[budget].append(error_diff)
                    target_diff_per_budget[budget].append(target_addr)

    for budget, target_diff in target_diff_per_budget.items():
        print(
            f"Probing budget: {budget}, Number of target diff: {len(target_diff)} ({round(len(target_diff) / len(eval.results_answer_subnets) * 100, 2)})"
        )

    return d_errors_per_budget, target_diff_per_budget


def get_first_vp_under_40() -> list:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    target_coordinates = {}
    for target in targets:
        target_coordinates[target["addr"]] = (target["lat"], target["lon"])

    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    geo_resolver_shortest_ping_index = []
    ref_shortest_ping_index = defaultdict(list)

    eval: EvalResults = load_pickle(
        path_settings.RESULTS_PATH
        / "tier5_evaluation/results__best_hostname_geo_score.pickle"
    )

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_ref_shortest_ping.pickle"
    )

    for target_addr, results_per_metric in eval.results_answer_subnets.items():
        # get ref info
        try:
            ref_shortest_ping_vp = ref_shortest_ping_results[target_addr][
                "ref_shortest_ping_vp"
            ]
        except KeyError:
            continue
        # ref_shortest_ping_index.append(ref_shortest_ping_vp["index"])

        # only consider ip addresses above 40 km in ref
        if ref_shortest_ping_vp["d_error"] > 40:
            continue

        # get target results
        target_lat, target_lon = target_coordinates[target_addr]
        for _, target_results in results_per_metric["result_per_metric"].items():
            ecs_vps = target_results["ecs_vps"]

            d_error = target_results["ecs_shortest_ping_vp_per_budget"][50]["d_error"]
            if not d_error > 40:
                continue

            ref_index = 0
            scores = []
            for index, (vp, score) in enumerate(ecs_vps):
                scores.append(score)

                if vp == ref_shortest_ping_vp["addr"]:
                    ref_index = index
                    break

                # target_subnet = get_prefix_from_ip(target_addr)
                # vp_subnet = get_prefix_from_ip(vp)

                # if vp_subnet == target_subnet:
                #     continue
                # vp_lat, vp_lon, _ = vps_coordinates[vp]
                # d = distance(target_lat, vp_lat, target_lon, vp_lon)

                # if d < 40:
                #     break

            if ref_index:
                #     logger.info(f"{ref_shortest_ping_vp['index']=}")
                #     logger.info(f"{ref_index=}")
                #     logger.info(f"{ecs_vps[:ref_index]=}\n")
                geo_resolver_shortest_ping_index.append(ref_index)
            #     logger.info(f"{np.std(scores)}")

    return geo_resolver_shortest_ping_index


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    probing_budgets = [50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.VPS_VPS_MESHED_PINGS_TABLE, removed_vps
    )
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    vps_country = get_vps_country(vps)

    logger.info("BGP prefix score geoloc evaluation")

    score_dir = path_settings.RESULTS_PATH / "tier5_evaluation"

    for score_file in score_dir.iterdir():

        if "results" in score_file.name:
            continue

        scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)

        logger.info(f"ECS evaluation for score:: {score_file}")

        output_file = (
            path_settings.RESULTS_PATH
            / f"tier5_evaluation/{'results' + str(score_file).split('scores')[-1]}"
        )

        results_answer_subnets = {}
        results_answer_subnets = ecs_dns_vp_selection_eval(
            targets=targets,
            vps_per_subnet=vps_per_subnet,
            subnet_scores=scores.score_answer_subnets,
            ping_vps_to_target=ping_vps_to_target,
            last_mile_delay=last_mile_delay,
            vps_coordinates=vps_coordinates,
            probing_budgets=probing_budgets,
            vps_country=None,
        )

        results = EvalResults(
            target_scores=scores,
            results_answers=None,
            results_answer_subnets=results_answer_subnets,
            results_answer_bgp_prefixes=None,
        )

        logger.info(f"output file:: {output_file}")

        dump_pickle(
            data=results,
            output_file=output_file,
        )


if __name__ == "__main__":
    compute_scores = False
    evaluation = True
    make_figure = True

    if compute_scores:
        compute_score()

    if evaluation:
        evaluate()

    if make_figure:
        # d_errors_per_budget, target_diff_per_budget = get_diff_error_per_budget()
        # x, y = ecdf(d_errors_per_budget[50])
        # plot_cdf(
        #     x=x,
        #     y=y,
        #     output_path="error_diff_wrongly_geolocated",
        #     x_label="Geolocation error difference (km)",
        #     y_label="CDF of targets",
        # )

        geo_resolver_shortest_ping_index = get_first_vp_under_40()

        x, y = ecdf(geo_resolver_shortest_ping_index)

        proportion_under_500 = get_proportion_under(x, y, 500)
        logger.info(f"{proportion_under_500=}")

        plot_cdf(
            x=x,
            y=y,
            output_path="first_vp_index_under_40",
            x_label="Shortest ping VP index (all VPs)",
            y_label="CDF of targets",
            x_lim=50,
        )
