"""plot figure 2, GeoResolver VP selection againts all VPs and 2ms threshold"""

import numpy as np

from loguru import logger
from collections import defaultdict, OrderedDict

from georesolver.clickhouse.queries import (
    load_vps,
    load_targets,
    get_pings_per_target_extended,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_d_errors_ref,
    get_d_errors_random,
    get_d_errors_georesolver,
    get_vp_selection_per_target,
)
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_csv, load_pickle, load_json
from georesolver.common.settings import PathSettings, ClickhouseSettings


path_settings = PathSettings()
ch_settings = ClickhouseSettings()

TARGETS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
HOSTNAME_FILE = path_settings.HOSTNAMES_GEORESOLVER
VPS_MAPPING_TABLE = "vps_ecs_mapping__2025_04_13"
RESULTS_PATH = path_settings.RESULTS_PATH / "figure_2_left_right"


def compute_score() -> None:
    """
    calculate redirection distance between VP DNS resolution and target's one for all targets
    """
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    hostnames = load_csv(HOSTNAME_FILE)
    target_subnets = [t["subnet"] for t in targets]
    vp_subnets = [v["subnet"] for v in vps]

    # load score similarity between vps and targets
    get_scores(
        output_path=RESULTS_PATH / "score.pickle",
        hostnames=hostnames,
        target_subnets=target_subnets,
        vp_subnets=vp_subnets,
        target_ecs_table=VPS_MAPPING_TABLE,
        vps_ecs_table=VPS_MAPPING_TABLE,
    )


def evaluation() -> None:
    """
    get vps ranking for future evaluation
    """
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    scores = load_pickle(RESULTS_PATH / "score.pickle")

    get_vp_selection_per_target(
        output_path=RESULTS_PATH / "vp_selection.pickle",
        scores=scores,
        targets=[t["addr"] for t in targets],
        vps=vps,
    )


def plot_figure_2_left() -> None:
    """
    plot figure 2 left, GeoResolver against all VPs and random selection
    on RIPE Atlas Anchors dataset
    """
    logger.info("** PLOT FIGURE 2 LEFT **")

    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )

    # load GeoResolver VPs ranking
    vp_selection_per_target = load_pickle(RESULTS_PATH / "vp_selection.pickle")

    cdfs = []

    # add reference (ALL VPs)
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))
    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    # add georesolver results
    d_errors = get_d_errors_georesolver(
        targets=[t["addr"] for t in targets],
        pings_per_target=pings_per_target,
        vp_selection_per_target=vp_selection_per_target,
        vps_coordinates=vps_coordinates,
    )

    # plot georesolver results
    x, y = ecdf(d_errors)
    cdfs.append((x, y, "GeoResolver"))
    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"GeoResolver:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"GeoResolver:: median_error={round(m_error, 2)} [km]")

    # add random VP selection
    d_errors_random = get_d_errors_random(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_random)
    cdfs.append((x, y, "Shortest ping, 50 random VPs"))
    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"Random VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Random VPs:: median_error={round(m_error, 2)} [km]")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="figure_2_left_georesolver_vs_all_vps",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def plot_figure_2_right() -> None:
    """
    plot figure 2 right: latency threshold vs. redirection distance
    """
    logger.info("** PLOT FIGURE 2 RIGHT **")

    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )

    cdfs = []
    d_errors_per_latencies = defaultdict(list)
    for target_addr, pings in pings_per_target.items():
        # get shortest ping using all vps
        vp_addr, _, min_rtt = min(pings, key=lambda x: x[-1])

        # get target/vps coordinates
        try:
            target = vps_coordinates[target_addr]
            vp = vps_coordinates[vp_addr]
        except KeyError:
            continue

        # get dst
        d_error = distance(target["lat"], vp["lat"], target["lon"], vp["lon"])

        # save d_errors function of the latency
        if min_rtt >= 0 and min_rtt < 1:
            d_errors_per_latencies[(0, 1)].append(d_error)
        if min_rtt >= 1 and min_rtt < 2:
            d_errors_per_latencies[(1, 2)].append(d_error)
        if min_rtt >= 2 and min_rtt < 4:
            d_errors_per_latencies[(2, 4)].append(d_error)

    # plot each latency threshold curves
    d_errors_per_latencies = OrderedDict(
        sorted(d_errors_per_latencies.items(), key=lambda x: x[0][0])
    )
    for latency_threshold, d_errors in d_errors_per_latencies.items():
        x, y = ecdf(d_errors)
        label = f"{latency_threshold[0]} <= rtt <{latency_threshold[1]}"
        cdfs.append((x, y, label))
        m_error = round(np.median(x), 2)
        proportion_of_ip = get_proportion_under(x, y)
        logger.info("##################################################")
        logger.info(f"{label}")
        logger.info(f"Random VPs:: <40km={round(proportion_of_ip, 2)}")
        logger.info(f"Random VPs:: median_error={round(m_error, 2)} [km]")
        logger.info("##################################################")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="figure_2_right_georesolver_vs_all_vps",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def main(
    do_compute_score: bool = True,
    do_evaluation: bool = True,
    do_plot: bool = True,
) -> None:
    """
    entry point for Figure 2 plot:
        - fig 2 left: plot GeoResolver vs. all VPs on RIPE Altas Anchors
        - fig 2 right: latency threshold vs. geolocation error
    """
    if do_compute_score:
        compute_score()

    if do_evaluation:
        evaluation()

    if do_plot:
        plot_figure_2_left()
        plot_figure_2_right()


if __name__ == "__main__":
    main()
