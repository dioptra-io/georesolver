import os

from pathlib import Path
from pyasn import pyasn
from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
    load_targets,
    load_vps,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_multiple_cdf,
    plot_ref,
    plot_ecs_shortest_ping,
)
from georesolver.evaluation.evaluation_ecs_geoloc_functions import (
    ecs_dns_vp_selection_eval,
)
from georesolver.evaluation.evaluation_score_functions import get_scores
from georesolver.common.utils import get_parsed_vps, EvalResults, TargetScores
from georesolver.common.files_utils import load_json, load_pickle, dump_pickle, load_csv
from georesolver.common.settings import PathSettings, ClickhouseSettings


path_settings = PathSettings()
ch_settings = ClickhouseSettings()

# set to True to use evaluation data
NEW_EVAL = True
if not NEW_EVAL:
    os.environ["CLICKHOUSE_DATABASE"] = ch_settings.CLICKHOUSE_DATABASE_EVAL


def compute_score(output_path: Path) -> None:
    """calculate score for each organization/ns pair"""
    targets_table = ch_settings.VPS_FILTERED_FINAL_TABLE
    vps_table = ch_settings.VPS_FILTERED_FINAL_TABLE
    targets_ecs_table = ch_settings.VPS_ECS_MAPPING_TABLE
    vps_ecs_table = ch_settings.VPS_ECS_MAPPING_TABLE

    selected_hostnames = load_csv(
        path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv"
    )

    score_config = {
        "targets_table": targets_table,
        "main_org_threshold": 0.0,
        "bgp_prefixes_threshold": 0.0,
        "vps_table": vps_table,
        "selected_hostnames": selected_hostnames,
        "targets_ecs_table": targets_ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": ["jaccard"],
        "answer_granularities": ["answer_subnets"],
        "output_path": output_path,
    }

    get_scores(score_config)


def evaluate(score_file: Path, output_file: Path, probing_parameter: list) -> None:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    # if output_file.exists():
    #     return

    logger.info(f"Running geresolver analysis from score file:: {score_file}")

    targets = load_targets(ch_settings.VPS_FILTERED_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(
        path_settings.REMOVED_VPS
        if NEW_EVAL
        else path_settings.DATASET / "imc2024_generated_files/removed_vps.json"
    )
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)
    last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    ping_vps_to_target = get_pings_per_target("anchors_ping", removed_vps)

    logger.info("Tier 5:: Distance error vs. VPs selection budget")
    scores: TargetScores = load_pickle(score_file)

    results_answer_subnets = ecs_dns_vp_selection_eval(
        targets=targets,
        vps_per_subnet=vps_per_subnet,
        subnet_scores=scores.score_answer_subnets,
        ping_vps_to_target=ping_vps_to_target,
        last_mile_delay=last_mile_delay,
        vps_coordinates=vps_coordinates,
        probing_budgets=probing_parameter,
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


def plot_d_error_per_budget(
    in_file: Path,
    output_path: str = "tier5_per_budget",
    metric_evaluated: str = "d_error",
    legend_pos: str = "lower right",
) -> None:
    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval: EvalResults = load_pickle(in_file)

    d_errors_per_budget = defaultdict(list)
    for _, results_per_metric in eval.results_answer_subnets.items():
        try:
            results = results_per_metric["result_per_metric"]["jaccard"]
            shortest_ping_vp_per_budget: dict = results[
                "ecs_shortest_ping_vp_per_budget"
            ]
        except KeyError:
            continue

        for budget, shortest_ping_vp in shortest_ping_vp_per_budget.items():
            d_errors_per_budget[budget].append(shortest_ping_vp[metric_evaluated])

    # get cdf for each budget/rank
    for budget, d_errors in d_errors_per_budget.items():
        x, y = ecdf(d_errors)
        all_cdfs.append((x, y, f"{budget} VPs" if budget > 1 else f"{budget} VP"))

    plot_multiple_cdf(
        cdfs=all_cdfs,
        output_path=output_path,
        metric_evaluated=metric_evaluated,
        legend_pos=legend_pos,
    )


def plot_d_error_per_rank(
    output_path: str = "tier5_per_rank",
    metric_evaluated: str = "d_error",
    legend_pos: str = "lower right",
) -> None:
    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval: EvalResults = load_pickle(
        path_settings.RESULTS_PATH / "tier5_evaluation/results__d_error_per_rank.pickle"
    )

    d_errors_per_budget = defaultdict(list)
    for _, target_results in eval.results_answer_subnets.items():
        try:
            results = target_results["result_per_metric"]["jaccard"]
            shortest_ping_vp_per_budget: dict = results[
                "ecs_shortest_ping_vp_per_budget"
            ]
        except KeyError:
            continue

        for budget, shortest_ping_vp in shortest_ping_vp_per_budget.items():
            d_errors_per_budget[budget].append(shortest_ping_vp[metric_evaluated])

    # get cdf for each budget/rank
    for budget, d_errors in d_errors_per_budget.items():
        x, y = ecdf(d_errors)
        all_cdfs.append((x, y, f"{budget[0]}:{budget[1]} VPs"))

    plot_multiple_cdf(
        cdfs=all_cdfs,
        output_path=output_path,
        metric_evaluated=metric_evaluated,
        legend_pos=legend_pos,
    )


def main() -> None:
    compute_scores = False
    evaluate_d_error_per_budget = False
    evaluate_d_error_per_rank = True
    make_figs = True

    base_path = path_settings.RESULTS_PATH / "tier5_evaluation/"

    if compute_scores:
        compute_score(
            output_path=base_path / f"scores{'_new' if NEW_EVAL else ''}.pickle"
        )
    if evaluate_d_error_per_budget:

        probing_parameter = [1, 10, 50, 100]
        evaluate(
            score_file=base_path / f"scores{'_new' if NEW_EVAL else ''}.pickle",
            output_file=base_path
            / f"results__d_error_per_budget{'_new' if NEW_EVAL else ''}.pickle",
            probing_parameter=probing_parameter,
        )

    if evaluate_d_error_per_rank:
        probing_parameter = [
            (0, 50),
            (50, 100),
            (100, 500),
            (500, 1_000),
            (1_000, 2_000),
            (2_000, 10_000),
        ]
        evaluate(
            score_file=base_path / f"scores{'_new' if NEW_EVAL else ''}.pickle",
            output_file=base_path
            / f"results__d_error_per_rank{'_new' if NEW_EVAL else ''}.pickle",
            probing_parameter=probing_parameter,
        )

    if make_figs:
        results: EvalResults = load_pickle(
            path_settings.RESULTS_PATH
            / f"tier5_evaluation/results__d_error_per_budget{'_new' if NEW_EVAL else ''}.pickle"
        )
        cdfs = plot_ecs_shortest_ping(
            results=results.results_answer_subnets,
            probing_budgets_evaluated=[1, 10, 50],
            metric_evaluated="rtt",
        )
        plot_multiple_cdf(
            cdfs=cdfs,
            output_path=f"tier5_evaluation{'_new' if NEW_EVAL else ''}",
            metric_evaluated="d_error",
        )


if __name__ == "__main__":
    main()
