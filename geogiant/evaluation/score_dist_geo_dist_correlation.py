from scipy.stats import pearsonr

from geogiant.common.files_utils import load_pickle
from geogiant.common.utils import EvalResults
from geogiant.common.settings import PathSettings
from geogiant.common.plot import (
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
        / "tier4_evaluation/results__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    )
    eval_results: EvalResults = load_pickle(path)

    # Look at the correlation between score and error
    elected_rtt_score_dist = []
    results = eval_results.results_answer_subnets
    for _, target_results_per_metric in results.items():
        for metric, target_results in target_results_per_metric[
            "result_per_metric"
        ].items():
            if not metric in ["jaccard"]:
                continue

            no_ping_vp = target_results["no_ping_vp"]

            elected_rtt = -1
            elected_score = 1 - no_ping_vp["score"]
            elected_d_error = no_ping_vp["d_error"]

            print(elected_rtt, elected_score, elected_d_error)

            elected_rtt_score_dist.append((elected_rtt, elected_score, elected_d_error))

    rtt_dist = [x[0] for x in elected_rtt_score_dist]
    score_dist = [x[1] for x in elected_rtt_score_dist]
    error_dist = [x[2] for x in elected_rtt_score_dist]
    corr, _ = pearsonr(score_dist, error_dist)
    print("Correlation", corr)
    marker_colors = [c[0] for c in colors_blind]

    fig, ax = plot_scatter_multiple(
        [score_dist],
        [error_dist],
        xmin=0,
        xmax=1,
        ymin=0,
        ymax=10000,
        xscale="linear",
        yscale="log",
        xlabel="Minimum redirection distance",
        ylabel="Distance error (km)",
        marker_size=[1],
        markers=markers,
        marker_colors=marker_colors,
    )
    homogenize_legend(ax, "lower right")
    ofile = path_settings.FIGURE_PATH / "error_score_correlation.pdf"
    ofile = path_settings.FIGURE_PATH / "error_score_correlation.png"

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
    for score_threshold, distance_per_threhsold in sorted(
        error_per_score_threshold.items()
    ):

        if not distance_per_threhsold:
            continue

        if score_threshold >= min_threshold:
            good_geoloc_40 = [x for x in distance_per_threhsold if x < 40]

            ratio_40 = len(good_geoloc_40) / len(distance_per_threhsold)
            coverage.append(len(distance_per_threhsold))

            print(score_threshold, ratio_40, len(distance_per_threhsold))
            Y_40.append(ratio_40)
            X.append(score_threshold)

    Y_coverage = [c / len(elected_rtt_score_dist) for c in coverage]
    Ys = [Y_40, Y_coverage]
    Xs = [X for _ in Ys]

    labels = ["Error < 40 km", "Error < 100 km", "Coverage"]
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
    ofile = path_settings.FIGURE_PATH / "bad_geolocation_threshold.png"
    plot_save(ofile, is_tight_layout=True)

    return


if __name__ == "__main__":

    compute_correlation()
