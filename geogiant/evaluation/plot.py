"""methods for plotting graph"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger
from collections import defaultdict, OrderedDict

from geogiant.common.files_utils import load_pickle, load_json
from geogiant.common.utils import EvalResults
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

font = {"weight": "bold", "size": 16}  #'family' : 'normal',
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
fontsize_axis = 17
font_size_alone = 14
matplotlib.rc("font", **font)

markers = ["o", "s", "v", "^"]
linestyles = ["-", "--", "-.", ":"]

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
        x_label = "geolocation error [km]"
    if metric_evaluated == "rtt":
        x_label = "shortest ping RTT [ms]"

    return x_label


def plot_multiple_cdf(
    cdfs: list,
    output_path: str,
    metric_evaluated: str,
    legend_outside: str = False,
    legend_pos: str = "upper left",
) -> None:

    fig, ax1 = plt.subplots(1, 1)

    for i, (x, y, label) in enumerate(cdfs):
        ax1.plot(x, y, label=label, color=colors_blind[i][1])

    x_label = get_x_label(metric_evaluated)
    ax1.grid(linestyle="dotted")
    ax1.set_xlabel(x_label, fontsize=fontsize_axis)
    ax1.set_ylabel("proportion of targets", fontsize=fontsize_axis)
    plot_minus_40()

    if legend_outside:
        plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
    else:
        plt.legend(loc=legend_pos, fontsize=8)

    plt.tight_layout()
    plt.xscale("log")
    plt.xlim(left=0.1)
    plt.ylim((0, 1))
    plt.savefig(
        path_settings.FIGURE_PATH / f"{output_path}_{metric_evaluated}.png",
        bbox_inches="tight",
    )
    plt.show()


def plot_ref(metric_evaluated: str) -> None:

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_ref_shortest_ping.pickle"
    )

    x, y = ecdf(
        [
            r["ref_shortest_ping_vp"][metric_evaluated]
            for r in ref_shortest_ping_results.values()
        ]
    )

    ref_cdf = (x, y, "Reference SP")

    return ref_cdf


def get_proportion_under(x, y, threshold: int = 40) -> int:
    for i, distance in enumerate(x):
        if distance > threshold:
            proportion_of_ip = y[i]
            break

    return proportion_of_ip


def plot_minus_40() -> None:
    x = [40, 40]
    y = [0, 1]

    # Plotting the line with dots
    plt.plot(x, y, linestyle="dotted", color="grey")


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
            label += f", answer {label_granularity}"
        if label_hostnames:
            label += f", {label_hostnames} hostnames"
        if label_orgs:
            label += f", {label_orgs} orgs"
        if len(score_metrics) > 1:
            label += f", {metric}"

        cdfs.append((x, y, label))

        m_error = round(
            np.median([r for r in cdf]),
            2,
        )

        logger.info(f"Zero ping:: {metric}, median_error={round(m_error, 2)} [km]")

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
        for budget, d_errors in d_error_per_budget[metric].items():
            if budget in probing_budgets_evaluated:
                x, y = ecdf([d for d in d_errors])
                m_error = round(np.median(d_errors), 2)
                label = f"ECS SP"

                if label_granularity:
                    label += f", {label_granularity}"
                if label_hostnames:
                    label += f", {label_hostnames} hostnames"
                if label_orgs:
                    label += f", {label_orgs} orgs"
                if len(score_metrics) > 1:
                    label += f", {metric}"
                if len(probing_budgets_evaluated) > 1:
                    label += f", {budget} VPs"
                if label_bgp_prefix:
                    label += f",  BGP threshold={label_bgp_prefix}"

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


def plot_per_granularity(
    score_config: dict,
    results: dict,
    score_metrics: list[str] = ["jaccard", "intersection"],
    probing_budgets_evaluated: list[int] = [10, 20, 50],
    granularity: str = "",
    metric_evaluated: str = "d_error",
    nb_orgs: int = 0,
    print_shortest_ping: bool = False,
) -> None:
    hostname_per_cdn = score_config["hostname_per_cdn"]
    total_hostnames = set()
    for hostnames in hostname_per_cdn.values():
        total_hostnames.update(hostnames)

    logger.info(f"{len(total_hostnames)=}")
    logger.info(f"nb targets:: {len(results)}")

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
        for budget, d_errors in d_error_per_budget[metric].items():
            if budget in probing_budgets_evaluated:
                x, y = ecdf([d for d in d_errors])
                m_error = round(np.median(d_errors), 2)
                label = f"ECS SP, {len(total_hostnames)} hostnames"

                if granularity:
                    label += f", {granularity}"
                if len(score_metrics) > 1:
                    label += f", {metric}"
                if nb_orgs != 0:
                    label += f", {nb_orgs}"

                cdfs.append((x, y, label))

                proportion_of_ip = get_proportion_under(x, y)

                logger.info(
                    f"ECS SP:: {metric}, <40km={round(proportion_of_ip, 2)}, {len(total_hostnames)} hostnames"
                )
                logger.info(
                    f"ECS SP:: {metric}, median_error={round(m_error, 2)} [km], {len(total_hostnames)} hostnames"
                )

    return cdfs


def tier1_plot(
    eval_dir: Path,
    granularities: list[str] = ["answer_subnets"],
    score_metrics=["jaccard"],
    probing_budgets_evaluated: list[int] = [50],
    metric_evaluated: str = "d_error",
    filter_org: list[str] = None,
) -> None:

    fig, ax1 = plt.subplots(1, 1)

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_ref_shortest_ping.pickle"
    )

    eval_files = []
    for file in eval_dir.iterdir():
        if "result" in file.name:
            eval_files.append(file.name)

    filtered_eval_files = {}
    if filter_org:
        for file in eval_files:
            for org in filter_org:
                if org in file:
                    nb_hostnames = file.split("__")[1].split("_")[0]
                    filtered_eval_files[nb_hostnames] = file

        eval_files = filtered_eval_files

    eval_files = sorted(eval_files.items(), key=lambda x: x[0])

    for nb_hostnames, file in eval_files:

        logger.info(f"{file=}")

        eval: EvalResults = load_pickle(eval_dir / file)

        score_config = eval.target_scores.score_config

        logger.info("Answer subnet granularity::")
        cdfs = plot_per_granularity(
            score_config=score_config,
            results=eval.results_answer_subnets,
            score_metrics=score_metrics,
            probing_budgets_evaluated=probing_budgets_evaluated,
            metric_evaluated=metric_evaluated,
        )
        for i, (x, y, label) in enumerate(cdfs):
            ax1.plot(x, y, label=label)

    ref_shortest_ping_m_d = get_median(
        ref_shortest_ping_results, "ref_shortest_ping_vp", metric_evaluated
    )

    x, y = ecdf(
        [
            r["ref_shortest_ping_vp"][metric_evaluated]
            for r in ref_shortest_ping_results.values()
        ]
    )
    ax1.plot(x, y, label=f"Reference SP")

    logger.info(
        f"Reference shortest ping:: median_error={round(ref_shortest_ping_m_d, 2)} [km]"
    )

    key = ""
    if key == "d_error":
        x_label = "geolocation error [km]"
    if key == "min_rtt":
        x_label = "shortest ping RTT [ms]"

    x_label = ""
    if metric_evaluated == "d_error":
        x_label = "geolocation error [km]"
    if metric_evaluated == "rtt":
        x_label = "shortest ping RTT [ms]"

    plt.xlabel(x_label)
    plt.ylabel("proportion of targets")
    plt.legend(loc="upper left", fontsize=8)
    plt.xscale("log")
    plt.grid()
    plt.savefig(path_settings.FIGURE_PATH / f"tier1_{metric_evaluated}_{org}.png")
    plt.show()


def plot_end_to_end_results(
    eval_dir: Path,
    granularities: list[str] = ["anser_bgp_prefixes"],
    probing_budgets_evaluated: list[int] = [50],
    score_metrics=["jaccard"],
    metric_evaluated: str = "d_error",
    imc_nb_vps: list[int] = [10, 100, 500],
    filter_org: list[str] = None,
    legend_outside: bool = False,
    legend_pos: str = "upper left",
) -> None:
    fig, ax1 = plt.subplots(1, 1)

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_ref_shortest_ping.pickle"
    )

    imc_baseline_results = load_json(
        path_settings.RESULTS_PATH / "round_based_algo_file.json"
    )

    eval_files = []
    for file in eval_dir.iterdir():
        if "result" in file.name:
            eval_files.append(file.name)

    filtered_eval_files = []
    if filter_org:
        for file in eval_files:
            for org in filter_org:
                if org in file:
                    nb_hostnames = file.split("__")[1].split("_")[0]
                    filtered_eval_files[nb_hostnames] = file

        eval_files = filtered_eval_files

    for file in eval_files:

        eval: EvalResults = load_pickle(
            path_settings.RESULTS_PATH / f"tier3_evaluation/{file}"
        )

        logger.info(f"{file=} loaded")

        score_config = eval.target_scores.score_config

        if "answer_bgp_prefixes" in granularities:
            logger.info("Answer BGP prefixes granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answer_bgp_prefixes,
                score_metrics=score_metrics,
                metric_evaluated=metric_evaluated,
                probing_budgets_evaluated=probing_budgets_evaluated,
            )

    ref_shortest_ping_m_d = get_median(
        ref_shortest_ping_results, "ref_shortest_ping_vp", metric_evaluated
    )

    x, y = ecdf(
        [
            r["ref_shortest_ping_vp"][metric_evaluated]
            for r in ref_shortest_ping_results.values()
        ]
    )
    ax1.plot(x, y, label=f"Reference SP")

    logger.info(
        f"Reference shortest ping:: median_error={round(ref_shortest_ping_m_d, 2)} [km]"
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
            m_error = round(
                np.median(
                    [d[index] for d in imc_baseline_results[nb_init_vps] if d[index]]
                ),
                2,
            )
            ax1.plot(
                x,
                y,
                label=f"IMC baseline SP, {nb_init_vps} init VPs",
            )

            logger.info(
                f"IMC baseline SP, {nb_init_vps} init VPs, median_error={round(m_error, 2)} [km]"
            )

    x_label = ""
    if metric_evaluated == "d_error":
        x_label = "geolocation error [km]"
    if metric_evaluated == "rtt":
        x_label = "shortest ping RTT [ms]"

    plt.xlabel(x_label)
    plt.ylabel("proportion of targets")

    plt.legend(loc=legend_pos, fontsize=8)
    if legend_outside:
        plt.legend(bbox_to_anchor=(1, 1))
    plt.xscale("log")
    plt.grid()
    plt.savefig(
        path_settings.FIGURE_PATH / f"end_to_end_{metric_evaluated}.png",
        bbox_inches="tight",
    )
    plt.show()


def score_metrics_and_granularity(
    eval_dir: Path,
    output_path: Path,
    metric_evaluated: str = "d_error",
    plot_zp: bool = True,
    legend_outside: bool = False,
) -> None:

    for file in eval_dir.iterdir():
        if "result" in file.name:
            eval_file = file
            break

    all_cdfs = []

    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval: EvalResults = load_pickle(eval_file)

    logger.info(f"{eval_file=} loaded")

    cdfs = plot_ecs_shortest_ping(
        results=eval.results_answer_subnets,
        score_metrics=[
            "intersection",
            "jaccard",
            "jaccard_scope_linear_weight",
            "intersection_scope_poly_weight",
        ],
        metric_evaluated=metric_evaluated,
        label_granularity="/24 subnet",
        plot_zp=plot_zp,
    )
    all_cdfs.extend(cdfs)

    cdfs = plot_ecs_shortest_ping(
        results=eval.results_answer_bgp_prefixes,
        score_metrics=[
            "intersection",
            "jaccard",
            "jaccard_scope_linear_weight",
            "jaccard_scope_poly_weight",
        ],
        metric_evaluated=metric_evaluated,
        label_granularity="BGP prefix",
        plot_zp=plot_zp,
    )
    all_cdfs.extend(cdfs)

    plot_multiple_cdf(all_cdfs, output_path, metric_evaluated, legend_outside)


def one_org_vs_many(
    eval_dir: Path,
    output_path: Path,
    metric_evaluated: str = "d_error",
) -> None:
    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval_files = []
    for file in eval_dir.iterdir():
        if "result" in file.name:
            eval_files.append(file)

    filtered_eval_files = defaultdict(list)
    for file in eval_files:
        nb_hostnames = file.name.split("__")[1].split("_")[0]
        nb_orgs = file.name.split("hostname_")[1].split("_")[0]
        filtered_eval_files[nb_orgs].append((nb_hostnames, file))
        eval_files = filtered_eval_files

    for org in filtered_eval_files:
        filtered_eval_files[org] = sorted(filtered_eval_files[org], key=lambda x: x[0])

    filtered_eval_files = OrderedDict(
        filtered_eval_files.items(),
        key=lambda x: x[0],
    )

    for nb_orgs in eval_files:
        for nb_hostnames, file in eval_files[nb_orgs]:
            eval: EvalResults = load_pickle(file)

            logger.info(f"{file=} loaded")

            cdfs = plot_ecs_shortest_ping(
                results=eval.results_answer_subnets,
                metric_evaluated=metric_evaluated,
                label_hostnames=nb_hostnames,
                label_orgs=nb_orgs,
            )
            all_cdfs.extend(cdfs)

    plot_multiple_cdf(all_cdfs, output_path, metric_evaluated)


def bgp_prefix_threshold(
    eval_dir: Path,
    output_path: Path,
    metric_evaluated: str = "d_error",
    plot_zp: bool = False,
    legend_outside: bool = False,
    legend_pos: str = "lower right",
) -> None:

    ordered_files = []
    for file in eval_dir.iterdir():
        if "result" in file.name:
            bgp_prefix_threshold = file.name.split("score_")[-1].split("_")[0]
            ordered_files.append((int(bgp_prefix_threshold), file))

    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)
    ordered_files = sorted(ordered_files, key=lambda x: x[0])
    for bgp_prefix_threshold, file in ordered_files:

        logger.info(f"{bgp_prefix_threshold=}")

        eval: EvalResults = load_pickle(file)

        logger.info(f"{file=} loaded")
        score_config = eval.target_scores.score_config
        hostname_per_cdn = score_config["hostname_per_cdn"]

        total_hostnames = set()
        for _, hostnames in hostname_per_cdn.items():
            total_hostnames.update(hostnames)

        cdfs = plot_ecs_shortest_ping(
            results=eval.results_answer_subnets,
            score_metrics=["jaccard"],
            metric_evaluated=metric_evaluated,
            label_hostnames=len(total_hostnames),
            label_bgp_prefix=bgp_prefix_threshold,
            plot_zp=plot_zp,
        )
        all_cdfs.extend(cdfs)

    plot_multiple_cdf(
        all_cdfs, output_path, metric_evaluated, legend_outside, legend_pos=legend_pos
    )


def plot_ripe_ip_map(
    geo_resolver_sp: list[tuple],
    ripe_ip_map_sp: list[tuple],
    output_path: Path,
) -> None:

    cdfs = []
    x, y = ecdf([sp for _, sp in geo_resolver_sp])
    cdfs.append((x, y, "GeoResolve SP"))

    x, y = ecdf([sp for _, sp in ripe_ip_map_sp])
    cdfs.append((x, y, "Single radius SP"))

    plot_multiple_cdf(cdfs, output_path=output_path, metric_evaluated="d_error")


if __name__ == "__main__":
    plot_end_to_end_results(
        eval_dir=path_settings.RESULTS_PATH / "tier3_evaluation",
        granularities=["answer_subnets"],
        score_metrics=["jaccard"],
        probing_budgets_evaluated=[10, 20, 50],
        imc_nb_vps=[10],
        metric_evaluated="d_error",
    )
