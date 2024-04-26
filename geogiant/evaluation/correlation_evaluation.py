import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from dataclasses import dataclass
from pyasn import pyasn
from loguru import logger

from geogiant.analysis.plot import ecdf
from geogiant.common.files_utils import load_json, load_pickle, load_csv
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix

from geogiant.common.settings import PathSettings

from plot_utils.plot import (
    plot_multiple_cdf,
    plot_save,
    homogenize_legend,
    plot_scatter_multiple,
    markers,
    colors_blind,
    plot_multiple,
)

path_settings = PathSettings()


def compute_correlation():
    # answer_granularity = "answer_bgp_prefix"
    top_n_vps = 50
    path = (
        path_settings.RESULTS_PATH
        / f"results_1M_hostnames_answer_bgp_prefix_max_bgp_prefix.pickle"
    )
    path = path_settings.RESULTS_PATH / f"evaluation_routers_2ms.pickle"
    eval_results = load_pickle(path)

    # Look at the correlation between score and error
    elected_rtt_score_dist = []
    for answer_granularity, top_n_vps_results in eval_results.items():
        for top_n_vps, results in top_n_vps_results.items():
            for target, target_result in results.items():
                elected_rtt = target_result["elected_rtt"]
                elected_score = target_result["elected_vp_score"]
                elected_d_error = target_result["elected_d_error"]
                ref_d_error = target_result["ref_d_error"]

                # Discard impossible results
                if 150 * elected_rtt < elected_d_error:
                    # Either VP badly geolocated or target badly geolocated
                    print("Bad geolocation", elected_rtt, elected_d_error)
                    continue
                elected_rtt_score_dist.append(
                    (elected_rtt, elected_score, elected_d_error)
                )

    score_dist = [x[1] for x in elected_rtt_score_dist]
    rtt_dist = [x[0] for x in elected_rtt_score_dist]
    error_dist = [x[2] for x in elected_rtt_score_dist]
    corr, _ = pearsonr(score_dist, error_dist)
    print("Correlation", corr)
    marker_colors = [c[0] for c in colors_blind]
    fig, ax = plot_scatter_multiple(
        [score_dist],
        [rtt_dist],
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=300,
        xscale="linear",
        yscale="log",
        xlabel="Maximum score",
        ylabel="RTT of the VP with the maximum score (ms)",
        marker_size=[1],
        markers=markers,
        marker_colors=marker_colors,
    )

    homogenize_legend(ax, "lower right")
    ofile = f"resources/correlation/rtt_score_correlation.pdf"

    plot_save(ofile, is_tight_layout=True)

    fig, ax = plot_scatter_multiple(
        [score_dist],
        [error_dist],
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=10000,
        xscale="linear",
        yscale="log",
        xlabel="Maximum score",
        ylabel="Distance error of the VP with the maximum score (km)",
        marker_size=[1],
        markers=markers,
        marker_colors=marker_colors,
    )
    homogenize_legend(ax, "lower right")
    ofile = f"resources/correlation/error_score_correlation.pdf"

    plot_save(ofile, is_tight_layout=True)
    # We want to plot the number of targets with high error for different thresholds
    error_per_score_threshold = {x / 100: [] for x in range(1, 100)}
    for elected_rtt, elected_score, elected_d_error in elected_rtt_score_dist:
        for score_threshold in error_per_score_threshold:
            if elected_score > score_threshold:
                error_per_score_threshold[score_threshold].append(elected_d_error)

    Y_40 = []
    Y_100 = []
    coverage = []
    X = []
    min_threshold = min(elected_rtt_score_dist, key=lambda x: x[1])[1]
    for score_threshold, errors in sorted(error_per_score_threshold.items()):
        if score_threshold >= min_threshold:
            bad_geolocation_100 = [x for x in errors if x > 100]
            bad_geolocation_40 = [x for x in errors if x > 40]
            ratio_40 = len(bad_geolocation_40) / len(errors)
            ratio_100 = len(bad_geolocation_100) / len(errors)
            coverage.append(len(errors))
            print(score_threshold, ratio_40, ratio_100, len(errors))
            Y_40.append(ratio_40)
            Y_100.append(ratio_100)
            X.append(score_threshold)

    Y_coverage = [c / len(elected_rtt_score_dist) for c in coverage]
    Ys = [Y_40, Y_100, Y_coverage]
    Xs = [X for _ in Ys]

    labels = ["Error > 40 km", "Error > 100 km", "Coverage"]
    colors = [colors_blind[i % len(colors_blind)][0] for i in range(len(Ys))]
    fig, ax = plot_multiple(
        Xs,
        Ys,
        xmin=min(X),
        xmax=1,
        ymin=0,
        ymax=1,
        xlabel="Metric",
        ylabel="Fraction of targets",
        xticks_labels=None,
        xscale="linear",
        yscale="linear",
        labels=labels,
        colors=colors,
    )

    homogenize_legend(ax, "upper left")
    ofile = f"resources/correlation/bad_geolocation_threshold.pdf"
    plot_save(ofile, is_tight_layout=True)

    return


if __name__ == "__main__":

    compute_correlation()
