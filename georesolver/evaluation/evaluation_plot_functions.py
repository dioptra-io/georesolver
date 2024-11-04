"""methods for plotting graph"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from matplotlib import collections as matcoll
import matplotlib.colors as mpl_colors
import scipy

# import statsmodels.api as sm
import pandas as pd
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from pathlib import Path
from loguru import logger
from collections import defaultdict, OrderedDict

from georesolver.common.files_utils import load_pickle, load_json
from georesolver.common.utils import EvalResults
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

font = {"weight": "bold", "size": 16}  #'family' : 'normal',
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
fontsize_axis = 17
font_size_alone = 14
matplotlib.rc("font", **font)

markers = ["o", "s", "v", "^"]
linestyles = ["-", "--", "-.", ":"]


def homogenize_legend(ax, legend_location, legend_size=14):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, Line2D):
            new_handles.append(h)
        elif isinstance(h, Polygon):
            new_handles.append(
                Line2D([], [], linestyle=h.get_linestyle(), color=h.get_edgecolor())
            )
    ax.legend(
        loc=legend_location,
        prop={"size": legend_size},
        handles=new_handles,
        labels=labels,
    )


colors_blind = [
    ["blue", (0, 114.0 / 255, 178.0 / 255)],
    ["reddish_purple", (204.0 / 255, 121.0 / 255, 167.0 / 255)],
    ["black", (0, 0, 0)],
    ["orange", (230.0 / 255, 159.0 / 255, 0)],
    ["sky_blue", (86.0 / 255, 180.0 / 255, 233.0 / 255)],
    ["vermillon", (213.0 / 255, 94.0 / 255, 0)],
    ["bluish_green", (0, 158.0 / 255, 115.0 / 255)],
    ["dark_green", (41.0 / 255, 94.0 / 255, 17.0 / 255)],
    ["yellow", (240.0 / 255, 228.0 / 255, 66.0 / 255)],
]


def ecdf(data: list, array: bool = True):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    if not array:
        return pd.DataFrame({"x": x, "y": y})
    else:
        return x, y


def get_error_bars(results: dict) -> tuple:
    """parse data for plot"""
    x = []
    y = []
    e = []
    for budget, (median_distance, deviation) in results.items():
        x.append(budget)
        y.append(median_distance)
        e.append(deviation)

    return x, y, e


def get_plot(results: dict) -> tuple[tuple, tuple]:
    """parse data for plot"""
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for budget, (m_d, w_m_d) in results.items():
        x1.append(budget)
        y1.append(m_d)

        x2.append(budget)
        y2.append(w_m_d)

    return (x1, y1), (x2, y2)


def plot_no_pings(results: dict, out_file: str) -> None:
    fig, ax1 = plt.subplots(1, 1)

    p1, p2 = get_plot(results["answers"])
    ax1.plot(p1[0], p1[1], label="frontend fingerprint")
    ax1.plot(p2[0], p2[1], label="w frontend fingerprint")

    p1, p2 = get_plot(results["subnet"])
    ax1.plot(p1[0], p1[1], label="subnet fingerprint")
    ax1.plot(p2[0], p2[1], label="w subnet fingerprint")

    p1, p2 = get_plot(results["bgp_prefix"])
    ax1.plot(p1[0], p1[1], label="bgp fingerprint")
    ax1.plot(p2[0], p2[1], label="w bgp fingerprint")

    plt.xlabel("Probing Budget [nb pings]")
    plt.ylabel("Median Error [km]")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid()
    plt.yscale("log")
    plt.title(f"Geolocation Error Function of Probing Budget", fontsize=13)
    plt.savefig(path_settings.FIGURE_PATH / out_file)
    plt.show()


def plot_median_error_per_finger_printing_method(results: dict, out_file: str) -> None:
    fig, ax1 = plt.subplots(1, 1)

    for granularity, result in results.items():
        x, y, _ = get_error_bars(result)
        ax1.plot(x, y, label=f"{granularity} fingerprint")

    plt.xlabel("Probing Budget [nb pings]")
    plt.ylabel("Median Error [km]")
    plt.legend(loc="upper right", fontsize=10)
    plt.grid()
    plt.yscale("log")
    plt.title(f"Geolocation Error Function of Probing Budget", fontsize=13)
    plt.savefig(path_settings.FIGURE_PATH / out_file)
    plt.show()


def get_median(target_results: dict, key: str, metric: str = "d_error") -> float:
    """return the median distance error"""
    return round(
        np.median([r[key][metric] for r in target_results.values()]),
        2,
    )


def get_x_label(metric_evaluated) -> str:
    x_label = ""
    if metric_evaluated == "d_error":
        x_label = "Geolocation error (km)"
    if metric_evaluated == "rtt":
        x_label = "RTT (ms)"

    return x_label


def plot_cdf(
    x: list,
    y: list,
    output_path: str,
    x_label: str,
    y_label: str,
    x_lim: int = 1,
    y_lim: int = 1,
    legend_outside: str = False,
    legend_pos: str = "upper left",
    legend_size: int = 10,
) -> None:

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(x, y, color=colors_blind[0][1])
    ax1.grid(linestyle="dotted")
    ax1.set_xlabel(x_label, fontsize=fontsize_axis)
    ax1.set_ylabel(y_label, fontsize=fontsize_axis)

    homogenize_legend(ax1, legend_pos, legend_size=legend_size)
    plt.tight_layout()
    plt.xscale("log")
    # plt.xlim(left=x_lim)
    plt.ylim((0, y_lim))
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_multiple_cdf(
    cdfs: list,
    output_path: str,
    metric_evaluated: str,
    legend_outside: str = False,
    legend_pos: str = "upper left",
    legend_size: int = 10,
    under_padding: int = 0,
    x_limit_left: int = 1,
) -> None:

    fig, ax1 = plt.subplots(1, 1)

    for i, (x, y, label) in enumerate(cdfs):
        if "under" in label:
            ax1.plot(x, y, label=label, color=colors_blind[i][1])

        else:
            ax1.plot(x, y, label=label, color=colors_blind[i][1])

    x_label = get_x_label(metric_evaluated)
    ax1.grid(linestyle="dotted")
    ax1.set_xlabel(x_label, fontsize=fontsize_axis)
    ax1.set_ylabel("CDF of targets", fontsize=fontsize_axis)

    if metric_evaluated == "d_error":
        plot_limit(
            limit=40, metric_evaluated=metric_evaluated, under_padding_d=under_padding
        )
    else:
        plot_limit(
            limit=2, metric_evaluated=metric_evaluated, under_padding_d=under_padding
        )
    if legend_outside:
        plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
    else:
        plt.legend(loc=legend_pos, fontsize=8)

    homogenize_legend(ax1, legend_pos, legend_size=legend_size)
    plt.tight_layout()
    plt.xscale("log")
    plt.xlim(left=x_limit_left)
    plt.ylim((0, 1))
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.png",
        bbox_inches="tight",
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.pdf",
        bbox_inches="tight",
    )
    plt.show()



def plot_imc(
    metric_evaluated: str = "d_error",
    imc_nb_vps: int = [5_00],
) -> list:
    cdfs = []

    imc_baseline_results = load_json(
        path_settings.RESULTS_PATH / "round_based_algo_file.json"
    )

    if metric_evaluated == "d_error":
        index = 1
    else:
        index = -1
    for nb_init_vps in imc_baseline_results:
        if int(nb_init_vps) in imc_nb_vps:
            x, y = ecdf(
                [d[index] for d in imc_baseline_results[nb_init_vps] if d[index]]
            )

            cdfs.append((x, y, f"IMC baseline SP, {nb_init_vps} init VPs"))
            m_error = round(
                np.median(
                    [d[index] for d in imc_baseline_results[nb_init_vps] if d[index]]
                ),
                2,
            )

            logger.info(
                f"IMC baseline SP, {nb_init_vps} init VPs, median_error={round(m_error, 2)} [km]"
            )

    return cdfs


def plot_ref(metric_evaluated: str) -> None:

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results__ref_shortest_ping.pickle"
    )

    x, y = ecdf(
        [
            r["ref_shortest_ping_vp"][metric_evaluated]
            for r in ref_shortest_ping_results.values()
        ]
    )

    ref_cdf = (x, y, "Shortest ping, all VPs")

    ref_under_40 = get_proportion_under(x, y)
    logger.info(f"Ref:: <40km {ref_under_40}%")
    return ref_cdf


def plot_random(metric_evaluated: str) -> None:

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_random_shortest_ping.pickle"
    )

    x, y = ecdf(
        [
            r["random_shortest_ping_vp"][metric_evaluated]
            for r in ref_shortest_ping_results.values()
        ]
    )

    ref_cdf = (x, y, "Shortest ping, 50 random VPs")

    return ref_cdf


def get_proportion_under(x, y, threshold: int = 40) -> int:
    proportion_of_ip = 1
    for i, distance in enumerate(x):
        if distance > threshold:
            proportion_of_ip = y[i]
            break

    return round(proportion_of_ip, 2)


def plot_limit(
    limit: float,
    metric_evaluated: str = "d_error",
    under_padding_d: int = 29,
    under_padding_rtt: int = 0.95,
) -> None:
    x = [limit, limit]
    y = [0, 1]

    # label = ""
    # if metric_evaluated == "d_error":
    #     label = f"under {limit} [km]"
 
    # if metric_evaluated == "rtt":
    #     label = f"under {limit} ms"

    # Plotting the line with dots
    plt.plot(x, y, linestyle="dotted", color="grey")
    plt.annotate(
        f"{x[0]} {'km' if metric_evaluated == 'd_error' else 'ms'}",
        xy=(
            (
                x[0] - under_padding_d
                if metric_evaluated == "d_error"
                else x[0] - under_padding_rtt
            ),
            1 - 0.1,
        ),
        size=12,
    )


def plot_zero_ping(
    results: dict,
    score_metrics: list[str] = ["jaccard"],
    metric_evaluated: str = "d_error",
    label_granularity: str = "",
    label_hostnames: int = None,
    label_orgs: int = None,
) -> list[tuple]:
    cdfs = []
    no_ping_per_metric = defaultdict(list)
    for _, target_results_per_metric in results.items():
        for metric, target_results in target_results_per_metric[
            "result_per_metric"
        ].items():
            if not metric in score_metrics:
                continue
            no_ping_per_metric[metric].append(
                target_results["no_ping_vp"][metric_evaluated]
            )

    for metric, cdf in no_ping_per_metric.items():
        x, y = ecdf([r for r in cdf])

        label = f"ZP"

        if label_granularity:
            label += f", {label_granularity}"
        if label_hostnames:
            label += f"{label_hostnames} hostnames"
        if label_orgs:
            label += f"\n{label_orgs} NS/org"
        if len(score_metrics) > 1:
            if metric == "jaccard":
                label += f"\nJaccard"
            if metric == "intersection":
                label += f"\nSzymkiewicz–Simpson"
            # else:
            #     label += f"\n{' '.join(metric.split('_'))}"

        cdfs.append((x, y, label))

        m_error = round(
            np.median([r for r in cdf]),
            2,
        )

        logger.info(f"Zero ping:: {metric}, median_error={round(m_error, 2)} [km]")

    return cdfs


def distance_error_vs_latency(
    results: dict,
    probing_budgets_evaluated: list[int] = [50],
    score_metrics: list[str] = ["jaccard"],
    latency_threshold: list[int] = [(0, 1)],
    distance_threshold: int = 40,
) -> list[tuple]:
    cdfs = []
    d_error_per_budget = defaultdict(dict)
    for _, target_results_per_metric in results.items():
        for metric, target_results in target_results_per_metric[
            "result_per_metric"
        ].items():
            if not metric in score_metrics:
                continue
            for budget, ecs_shortest_ping_vp in target_results[
                "ecs_shortest_ping_vp_per_budget"
            ].items():
                try:
                    d_error_per_budget[metric][budget].append(
                        (ecs_shortest_ping_vp["d_error"], ecs_shortest_ping_vp["rtt"])
                    )
                except KeyError:
                    d_error_per_budget[metric][budget] = [
                        (ecs_shortest_ping_vp["d_error"], ecs_shortest_ping_vp["rtt"])
                    ]

    for metric in d_error_per_budget:
        for budget, target_results in d_error_per_budget[metric].items():
            if budget in probing_budgets_evaluated:
                d_error_under_latency = []
                for d_error, latency in target_results:
                    if (
                        latency >= latency_threshold[0]
                        and latency < latency_threshold[1]
                    ):
                        d_error_under_latency.append(d_error)

                x, y = ecdf(d_error_under_latency)
                cdfs.append(
                    (x, y, f"{latency_threshold[0]} <= rtt < {latency_threshold[1]} ms")
                )

                proportion_of_ip = get_proportion_under(x, y, 40)

                print(
                    f"IP addresses={len(d_error_under_latency)}; <{distance_threshold}km; {latency_threshold[0]}<= rtt <={latency_threshold[1]}ms = {round(proportion_of_ip, 2)}"
                )

    return cdfs


def geo_resolver_cdfs(results: dict, metric_evaluated: str, label: list[str]) -> None:
    cdfs = []
    data = []
    for _, target_results_per_metric in results.items():
        try:
            ecs_shortest_ping_vp = target_results_per_metric["result_per_metric"][
                "jaccard"
            ]["ecs_shortest_ping_vp_per_budget"][50]
        except KeyError:
            continue

        if ecs_shortest_ping_vp[metric_evaluated]:
            data.append(ecs_shortest_ping_vp[metric_evaluated])

    x, y = ecdf(data)
    cdfs.append((x, y, label))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)
    logger.info(f"ECS SP:: {label}: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"ECS SP:: {label}: median_error={round(m_error, 2)} [km]")

    return cdfs


def plot_ecs_shortest_ping(
    results: dict,
    probing_budgets_evaluated: list[int] = [50],
    score_metrics: list[str] = ["jaccard"],
    metric_evaluated: str = "d_error",
    label_granularity: str = "",
    label_orgs: int = None,
    label_hostnames: int = None,
    label_bgp_prefix: int = None,
    label_nb_hostname_per_org_ns: int = None,
    plot_zp: bool = False,
) -> list[tuple]:
    cdfs = []
    d_error_per_budget = defaultdict(dict)
    for _, target_results_per_metric in results.items():
        for metric, target_results in target_results_per_metric[
            "result_per_metric"
        ].items():
            if not metric in score_metrics:
                continue
            for budget, ecs_shortest_ping_vp in target_results[
                "ecs_shortest_ping_vp_per_budget"
            ].items():
                try:
                    d_error_per_budget[metric][budget].append(
                        ecs_shortest_ping_vp[metric_evaluated]
                    )
                except KeyError:
                    d_error_per_budget[metric][budget] = [
                        ecs_shortest_ping_vp[metric_evaluated]
                    ]

    for metric in d_error_per_budget:
        d_error_per_budget[metric] = sorted(
            d_error_per_budget[metric].items(), key=lambda x: x[0], reverse=True
        )
        for budget, d_errors in d_error_per_budget[metric]:
            if budget in probing_budgets_evaluated:
                x, y = ecdf([d for d in d_errors])
                m_error = round(np.median(d_errors), 2)

                label = "GeoResolver"

                if label_granularity:
                    label += f", {label_granularity}"
                if label_hostnames:
                    label += f", {label_hostnames} hostnames"
                if label_orgs:
                    label += f"\n{label_orgs} NS/org"
                if len(score_metrics) > 1:
                    if metric == "jaccard":
                        label += f"\nJaccard"
                    if metric == "intersection":
                        label += f"\nSzymkiewicz–Simpson"
                    if metric == "jaccard_scope_linear_weight":
                        label += f"\nJaccard scope \nlinear weight"

                    if metric == "jaccard_scope_poly_weight":
                        label += f"\nJaccard scope \npoly weight"

                    if metric == "intersection_scope_linear_weight":
                        label += f"\nSzymkiewicz–Simpson scope \nlinear weight"

                    if metric == "intersection_scope_poly_weight":
                        label += f"\nSzymkiewicz–Simpson scope \npoly weight"
                if len(probing_budgets_evaluated) > 1:
                    label += f", {budget} VPs"
                if label_bgp_prefix:
                    label += f"\nBGP threshold={label_bgp_prefix}"
                if label_nb_hostname_per_org_ns:
                    label += f"\n{label_nb_hostname_per_org_ns} hostnames per NS/org"

                # if (
                #     metric == "jaccard"
                #     and len(probing_budgets_evaluated) > 1
                #     and label_granularity
                # ):
                #     label += f" (GeoResolver {budget} VPs)"

                # elif metric == "jaccard" and len(probing_budgets_evaluated) > 1:
                #     label += f"GeoResolver {budget} VPs"
                # else:
                #     label += "GeoResolver"

                cdfs.append((x, y, label))

                proportion_of_ip = get_proportion_under(x, y)
                logger.info(
                    f"ECS SP:: {metric}, <40km={round(proportion_of_ip, 2)}, {label_hostnames} hostnames"
                )
                logger.info(
                    f"ECS SP:: {metric}, median_error={round(m_error, 2)} [km], {label_hostnames} hostnames"
                )

    if plot_zp:
        zp_cdfs = plot_zero_ping(
            results=results,
            score_metrics=score_metrics,
            metric_evaluated=metric_evaluated,
            label_granularity=label_granularity,
            label_hostnames=label_hostnames,
            label_orgs=label_orgs,
        )
        cdfs.extend(zp_cdfs)

    return cdfs


def plot_end_to_end_results(
    score_file: Path,
    output_path: str,
    probing_budgets_evaluated: list[int] = [50],
    score_metrics=["jaccard"],
    metric_evaluated: str = "d_error",
    legend_pos="upper left",
) -> None:
    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval: EvalResults = load_pickle(score_file)

    logger.info(f"{score_file=} loaded")

    cdfs = plot_ecs_shortest_ping(
        results=eval.results_answer_subnets,
        score_metrics=score_metrics,
        metric_evaluated=metric_evaluated,
        probing_budgets_evaluated=probing_budgets_evaluated,
    )
    all_cdfs.extend(cdfs)

    cdf_imc = plot_random(metric_evaluated=metric_evaluated)
    all_cdfs.append(cdf_imc)

    plot_multiple_cdf(all_cdfs, output_path, metric_evaluated, False, legend_pos)


def plot_routers(
    geo_resolver_sp: dict[list[tuple]],
    ref_sp: list[tuple],
    random_sp: list[tuple],
    output_path: Path,
) -> None:

    cdfs = []

    x, y = ecdf([sp for _, sp in ref_sp])
    m = round(np.mean(x), 2)
    proportion = get_proportion_under(x, y, threshold=2)
    logger.info("Reference::")
    logger.info(f"Nb targets:: {len(x)}")
    logger.info(f"Proportion under 2ms: {proportion} ms")
    logger.info(f"Median latency: {m} ms")
    cdfs.append((x, y, "Shortest ping, all VPs"))

    for budget, results in geo_resolver_sp.items():
        if budget not in [50]:
            continue
        x, y = ecdf([sp for _, _, sp in results])
        cdfs.append(
            (
                x,
                y,
                f"GeoResolver",
            )
        )

        m = round(np.mean(x), 2)
        proportion = get_proportion_under(x, y, threshold=2)
        logger.info(f"Nb targets:: {len(x)}")
        logger.info(f"Proportion under 2ms: {proportion} ms")
        logger.info(f"Median latency: {m} ms")

    x, y = ecdf([sp for _, sp in random_sp])
    m = round(np.mean(x), 2)
    proportion = get_proportion_under(x, y, threshold=2)
    logger.info("Random 50 VPs::")
    logger.info(f"Nb targets:: {len(x)}")
    logger.info(f"Proportion under 2ms: {proportion} ms")
    logger.info(f"Median latency: {m} ms")
    cdfs.append((x, y, "Random shortest ping, 50 VPs"))

    plot_multiple_cdf(
        cdfs,
        output_path=output_path,
        metric_evaluated="rtt",
        legend_pos="lower right",
        legend_size=9,
    )


def plot_ripe_ip_map(
    geo_resolver_sp: dict[list[tuple]],
    ripe_ip_map_sp: list[tuple],
    ref_sp: list[tuple],
    random_sp: list[tuple],
    output_path: Path,
) -> None:

    cdfs = []

    x, y = ecdf([sp for _, sp in ref_sp])
    m = round(np.mean(x), 2)
    proportion = get_proportion_under(x, y, threshold=2)
    logger.info("Reference::")
    logger.info(f"Nb targets:: {len(x)}")
    logger.info(f"Proportion under 2ms: {proportion} ms")
    logger.info(f"Median latency: {m} ms")
    cdfs.append((x, y, "Shortest ping, all VPs"))

    for budget, results in geo_resolver_sp.items():
        if budget not in [50]:
            continue
        x, y = ecdf([sp for _, _, sp in results])
        cdfs.append(
            (
                x,
                y,
                f"GeoResolver",
            )
        )

        m = round(np.mean(x), 2)
        proportion = get_proportion_under(x, y, threshold=2)
        logger.info(f"Nb targets:: {len(x)}")
        logger.info(f"Proportion under 2ms: {proportion} ms")
        logger.info(f"Median latency: {m} ms")

    x, y = ecdf([sp for _, sp in ripe_ip_map_sp])
    m = round(np.mean(x), 2)
    proportion = get_proportion_under(x, y, threshold=2)
    logger.info("Single radius::")
    logger.info(f"Nb targets:: {len(x)}")

    logger.info(f"Proportion under 2ms: {proportion} ms")
    logger.info(f"Median latency: {m} ms")
    cdfs.append((x, y, "Single-radius, AVG 500 VPs"))

    x, y = ecdf([sp for _, sp in random_sp])
    m = round(np.mean(x), 2)
    proportion = get_proportion_under(x, y, threshold=2)
    logger.info("Random 50 VPs::")
    logger.info(f"Nb targets:: {len(x)}")
    logger.info(f"Proportion under 2ms: {proportion} ms")
    logger.info(f"Median latency: {m} ms")
    cdfs.append((x, y, "Random shortest ping, 50 VPs"))

    plot_multiple_cdf(
        cdfs,
        output_path=output_path,
        metric_evaluated="rtt",
        legend_pos="lower right",
        legend_size=8,
    )


def plot_internet_scale(
    geo_resolver_sp: dict[list[tuple]],
    output_path: Path,
) -> None:

    cdfs = []

    x, y = ecdf([sp for _, sp in geo_resolver_sp])
    cdfs.append((x, y, f"GeoResolver"))

    m = round(np.mean(x), 2)
    proportion = get_proportion_under(x, y, threshold=2)
    logger.info(f"Nb targets:: {len(x)}")
    logger.info(f"Proportion under 2ms: {proportion} ms")
    logger.info(f"Median latency: {m} ms")

    plot_multiple_cdf(
        cdfs,
        output_path=output_path,
        metric_evaluated="rtt",
        legend_pos="lower right",
        legend_size=9,
    )


if __name__ == "__main__":
    plot_d_error_vs_latency(
        score_file=path_settings.RESULTS_PATH
        / "tier3_evaluation/results__best_hostname_geo_score.pickle",
        output_path="d_error_vs_latency",
    )
