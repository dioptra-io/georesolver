import numpy as np

from loguru import logger

from georesolver.clickhouse.queries import (
    load_vps,
    load_targets,
    get_pings_per_target_extended,
)
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_d_errors_ref,
    get_d_errors_georesolver,
    get_vp_selection_per_target,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.common.files_utils import load_json, load_pickle
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

TARGETS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_ECS_TABLE = "vps_ecs_mapping__2025_04_13"
RESULTS_PATH = path_settings.RESULTS_PATH / "figure_3"


def load_hostnames() -> set:
    """return a list of selected hostnames"""
    best_hostnames_per_org_per_ns = load_json(
        path_settings.HOSTNAME_FILES / "hostname__20_BGP_3_org_ns.json",
    )

    selected_hostnames = set()
    for _, hostnames_per_org in best_hostnames_per_org_per_ns.items():
        for _, hostnames in hostnames_per_org.items():
            selected_hostnames.update(hostnames)

    return selected_hostnames


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    selected_hostnames = load_hostnames()
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)

    # calculate score for diff granularities
    answer_granularities = [
        "answers",
        "answer_subnets",
        "answer_bgp_prefixes",
    ]
    for granularity in answer_granularities:
        output_path = RESULTS_PATH / f"scores__granularity_{granularity}.pickle"

        get_scores(
            output_path=output_path,
            hostnames=selected_hostnames,
            target_subnets=[t["subnet"] for t in targets],
            vp_subnets=[v["subnet"] for v in vps],
            target_ecs_table=VPS_ECS_TABLE,
            vps_ecs_table=VPS_ECS_TABLE,
            answer_granularity=granularity,
            score_metric="jaccard",
        )

    # we only calculate scores with jaccard distance when considering
    # source scope length, otherwise too long and not particularly interesting
    score_metrics = [
        "intersection",
        "jaccard",
        # "intersection_scope_linear_weight",
        # "intersection_scope_poly_weight",
        # "intersection_scope_exp_weight",
        "jaccard_scope_linear_weight",
        "jaccard_scope_poly_weight",
        "jaccard_scope_exp_weight",
    ]
    for score_metric in score_metrics:
        output_path = RESULTS_PATH / f"scores__metrics_{score_metric}.pickle"
        logger.info(f"Calculating scores for {score_metric=}; {output_path=}")

        get_scores(
            output_path=output_path,
            hostnames=selected_hostnames,
            target_subnets=[t["subnet"] for t in targets],
            vp_subnets=[v["subnet"] for v in vps],
            target_ecs_table=TARGET_ECS_TABLE,
            vps_ecs_table=VPS_ECS_TABLE,
            answer_granularity="answer_subnets",
            score_metric=score_metric,
        )


def evaluation() -> None:
    """calculate distance error and latency for each score"""
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)

    for score_file in RESULTS_PATH.iterdir():

        if "scores" not in score_file.name:
            continue

        output_file = (
            RESULTS_PATH / f"{'results' + str(score_file).split('scores')[-1]}"
        )

        scores = load_pickle(score_file)

        get_vp_selection_per_target(
            output_path=output_file,
            scores=scores,
            targets=[t["addr"] for t in targets],
            vps=vps,
        )


def plot_figure_3_left() -> None:
    """output figure of the figure 3 left of GeoResolver's paper"""
    logger.info("** PLOT FIGURE 3 LEFT **")

    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    probing_budgets = [50, 1]
    result_files = {
        "results__granularity_answers.pickle": "Server IP address",
        "results__granularity_answer_subnets.pickle": "/24",
        "results__granularity_answer_bgp_prefixes.pickle": "BGP",
    }

    cdfs = []

    # add reference (ALL VPs)
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))
    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    for probing_budget in probing_budgets:
        for result_file, granularity in result_files.items():

            vp_selection_per_target = load_pickle(RESULTS_PATH / result_file)
            d_errors = get_d_errors_georesolver(
                targets=[t["addr"] for t in targets],
                pings_per_target=pings_per_target,
                vp_selection_per_target=vp_selection_per_target,
                vps_coordinates=vps_coordinates,
                probing_budget=probing_budget,
            )

            x, y = ecdf(d_errors)
            m_error = round(np.median(x), 2)
            proportion_of_ip = get_proportion_under(x, y)
            if granularity == "/24" and probing_budget == 50:
                label = f"{granularity}, {probing_budget} VPs (GeoResolver)"
            else:
                label = f"{granularity}, {probing_budget} {'VPs' if probing_budget > 1 else 'VP'}"
            cdfs.append((x, y, label))

            logger.info("##################################################")
            logger.info(f"{label}")
            logger.info(f"frac <40km={round(proportion_of_ip, 2)}")
            logger.info(f"median_error={round(m_error, 2)} [km]")
            logger.info("##################################################")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="figure_3_left_distance_error_vs_granylarity",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def plot_figure_3_center() -> None:
    """output figure of the figure 3 center of GeoResolver's paper"""
    logger.info("** PLOT FIGURE 3 RIGHT **")

    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    probing_budgets = [50, 1]
    result_files = {
        "results__metrics_jaccard.pickle": "Jaccard",
        "results__metrics_intersection.pickle": "Szymkiewicz-Simpson",
    }

    cdfs = []

    # add reference (ALL VPs)
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))
    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    for probing_budget in probing_budgets:
        for result_file, score_metric in result_files.items():

            vp_selection_per_target = load_pickle(RESULTS_PATH / result_file)
            d_errors = get_d_errors_georesolver(
                targets=[t["addr"] for t in targets],
                pings_per_target=pings_per_target,
                vp_selection_per_target=vp_selection_per_target,
                vps_coordinates=vps_coordinates,
                probing_budget=probing_budget,
            )

            x, y = ecdf(d_errors)
            m_error = round(np.median(x), 2)
            proportion_of_ip = get_proportion_under(x, y)
            if score_metric == "Jaccard" and probing_budget == 50:
                label = f"{score_metric}, {probing_budget} VPs (GeoResolver)"
            else:
                label = f"{score_metric}, {probing_budget} {'VPs' if probing_budget > 1 else 'VP'}"
            cdfs.append((x, y, label))

            logger.info("##################################################")
            logger.info(f"{label}")
            logger.info(f"frac <40km={round(proportion_of_ip, 2)}")
            logger.info(f"median_error={round(m_error, 2)} [km]")
            logger.info("##################################################")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="figure_3_center_distance_error_vs_score_metric",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def plot_figure_3_right() -> None:
    """output figure of the figure 3 right of GeoResolver's paper"""
    logger.info("** PLOT FIGURE 3 RIGHT **")

    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    probing_budgets = [50, 1]
    result_files = {
        "results__metrics_jaccard.pickle": "Unweighted",
        "results__metrics_jaccard_scope_linear_weight.pickle": "Linear weight",
        "results__metrics_jaccard_scope_poly_weight.pickle": "Square weight",
    }

    cdfs = []

    # add reference (ALL VPs)
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))
    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    for probing_budget in probing_budgets:
        for result_file, weight in result_files.items():

            vp_selection_per_target = load_pickle(RESULTS_PATH / result_file)
            d_errors = get_d_errors_georesolver(
                targets=[t["addr"] for t in targets],
                pings_per_target=pings_per_target,
                vp_selection_per_target=vp_selection_per_target,
                vps_coordinates=vps_coordinates,
                probing_budget=probing_budget,
            )

            x, y = ecdf(d_errors)
            m_error = round(np.median(x), 2)
            proportion_of_ip = get_proportion_under(x, y)
            if weight == "Unweighted" and probing_budget == 50:
                label = f"{weight}, {probing_budget} VPs (GeoResolver)"
            else:
                label = f"{weight}, {probing_budget} {'VPs' if probing_budget > 1 else 'VP'}"
            cdfs.append((x, y, label))

            logger.info("##################################################")
            logger.info(f"{label}")
            logger.info(f"frac <40km={round(proportion_of_ip, 2)}")
            logger.info(f"median_error={round(m_error, 2)} [km]")
            logger.info("##################################################")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="figure_3_center_distance_error_vs_score_metric",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def main(
    do_compute_score=False,
    do_evaluation=False,
    do_plot=True,
) -> None:
    """
    plot figure 3:
        - left: GeoResolver accuracy for different DNS redirection granularity
        - center: GeoResolver accuracy for different DNS redirection distance evaluation
        - right: GeoResolver accuracy when taking in account source scope
    """
    if do_compute_score:
        compute_score()

    if do_evaluation:
        evaluation()

    if do_plot:
        plot_figure_3_left()
        plot_figure_3_center()
        plot_figure_3_right()


if __name__ == "__main__":
    main()
