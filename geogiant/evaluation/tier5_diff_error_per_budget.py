import os

from collections import defaultdict
from loguru import logger
from pyasn import pyasn

from geogiant.clickhouse.queries import (
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
from geogiant.evaluation.evaluation_plot_functions import ecdf, plot_cdf, get_proportion_under
from geogiant.evaluation.evaluation_ecs_geoloc_functions import ecs_dns_vp_selection_eval
from geogiant.evaluation.evaluation_score_functions import get_scores
from geogiant.common.files_utils import load_json, load_pickle, dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings


path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["CLICKHOUSE_DATABASE"] = ch_settings.CLICKHOUSE_DATABASE_EVAL


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    targets_table = ch_settings.VPS_FILTERED_TABLE
    vps_table = ch_settings.VPS_FILTERED_TABLE
    targets_ecs_table = ch_settings.VPS_ECS_MAPPING_TABLE
    vps_ecs_table = ch_settings.VPS_ECS_MAPPING_TABLE

    selected_hostnames_per_cdn_per_ns = load_json(
        path_settings.HOSTNAME_FILES
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


def plot_diff_error_per_budget() -> tuple[dict, dict]:

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

    x, y = ecdf(d_errors_per_budget[50])
    plot_cdf(
        x=x,
        y=y,
        output_path="error_diff_wrongly_geolocated",
        x_label="Geolocation error difference (km)",
        y_label="CDF of targets",
    )


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    probing_budgets = [50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        ch_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )
    targets = load_targets(ch_settings.VPS_FILTERED_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)

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
        plot_diff_error_per_budget()
