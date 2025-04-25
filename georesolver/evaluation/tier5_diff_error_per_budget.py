import numpy as np

from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import (
    get_pings_per_target_extended,
    load_vps,
    load_targets,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_vp_selection_per_target,
)
from georesolver.common.files_utils import load_json, load_pickle, load_csv
from georesolver.common.utils import get_d_errors_georesolver, get_d_errors_ref
from georesolver.common.settings import PathSettings, ClickhouseSettings


path_settings = PathSettings()
ch_settings = ClickhouseSettings()

TARGETS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
TARGETS_ECS_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE
VPS_ECS_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE
RESULTS_PATH = path_settings.RESULTS_PATH / "tier5_evaluation"


def plot_per_budget() -> None:
    """plot georesolver results function of the probing budget"""
    cdfs = []
    budgets = [500, 100, 50, 10, 1]
    removed_vps = load_json(path_settings.REMOVED_VPS)
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )

    # load score similarity between vps and targets
    scores = get_scores(
        output_path=RESULTS_PATH / "score.pickle",
        hostnames=load_csv(path_settings.HOSTNAMES_GEORESOLVER),
        target_subnets=[t["addr"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=ch_settings.VPS_ECS_MAPPING_TABLE,
    )

    vp_selection_per_target = get_vp_selection_per_target(
        output_path=RESULTS_PATH / "vp_selection.pickle",
        scores=scores,
        targets=pings_per_target.keys(),
        vps=vps,
    )

    # add reference
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    # add georesolver results
    for budget in budgets:

        d_errors = get_d_errors_georesolver(
            pings_per_target=pings_per_target,
            vp_selection_per_target=vp_selection_per_target,
            vps_coordinates=vps_coordinates,
            probing_budget=budget,
        )

        # get label
        if budget == 1:
            label = "1 VP"
        elif budget == 50:
            label = "50 VPs (GeoResolver)"
        else:
            label = f"{budget} VPs"

        # plot georesolver results
        x, y = ecdf(d_errors)
        cdfs.append((x, y, label))

        m_error = round(np.median(x), 2)
        proportion_of_ip = get_proportion_under(x, y)

        logger.info(f"GeoResolver:: {budget=}: <40km={round(proportion_of_ip, 2)}")
        logger.info(f"GeoResolver:: {budget=}: median_error={round(m_error, 2)} [km]")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="tier5_per_budget",
        metric_evaluated="d_error",
    )


def plot_d_error_per_rank() -> None:
    """
    plot georesolver per VPs rank batch (ex: first 50,
    vps ranked from 50 to 100, etc.)
    """
    cdfs = []
    ranks = [
        (0, 50),
        (50, 100),
        (100, 500),
        (500, 1_000),
        (1_000, 2_000),
        (2_000, 10_000),
    ]
    removed_vps = load_json(path_settings.REMOVED_VPS)
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )

    # load score similarity between vps and targets
    scores = get_scores(
        output_path=RESULTS_PATH / "score.pickle",
        hostnames=load_csv(path_settings.HOSTNAMES_GEORESOLVER),
        target_subnets=[t["addr"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=ch_settings.VPS_ECS_MAPPING_TABLE,
    )

    vp_selection_per_target = get_vp_selection_per_target(
        output_path=RESULTS_PATH / "vp_selection.pickle",
        scores=scores,
        targets=pings_per_target.keys(),
        vps=vps,
    )

    # add reference
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    # add georesolver results
    for rank in ranks:

        d_errors = get_d_errors_georesolver(
            pings_per_target=pings_per_target,
            vp_selection_per_target=vp_selection_per_target,
            vps_coordinates=vps_coordinates,
            probing_budget=rank,
        )

        # get label
        if rank[0] == 0:
            label = f"{rank[0]}:{rank[1]} VPs"
        else:
            label = (
                f"{rank[0]}:{rank[1]} VPs" + " (GeoResolver)" if rank[0] == 0 else ""
            )

        # plot georesolver results
        x, y = ecdf(d_errors)
        cdfs.append((x, y, label))

        m_error = round(np.median(x), 2)
        proportion_of_ip = get_proportion_under(x, y)

        logger.info(f"GeoResolver:: {rank=}: <40km={round(proportion_of_ip, 2)}")
        logger.info(f"GeoResolver:: {rank=}: median_error={round(m_error, 2)} [km]")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="tier5_per_rank",
        metric_evaluated="d_error",
    )


def main() -> None:
    do_plot_per_budget = True
    do_plot_per_rank = False

    if do_plot_per_budget:
        plot_per_budget()

    if do_plot_per_rank:
        probing_parameter = [
            (0, 50),
            (50, 100),
            (100, 500),
            (500, 1_000),
            (1_000, 2_000),
            (2_000, 10_000),
        ]
        plot_per_rank(
            score_file=base_path / f"scores.pickle",
            output_file=base_path / f"results__d_error_per_rank.pickle",
            probing_parameter=probing_parameter,
        )


if __name__ == "__main__":
    main()
