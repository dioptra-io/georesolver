"""methods for plotting graph"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from loguru import logger
from collections import defaultdict

from geogiant.common.files_utils import load_pickle, load_json
from geogiant.common.utils import EvalResults
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


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


def plot_per_granularity(
    ax1,
    score_config: dict,
    results: dict,
    score_metrics: list[str] = ["jaccard", "intersection"],
    probing_budgets_evaluated: list[int] = [10, 20, 50],
    granularity: str = "",
    metric_evaluated: str = "d_error",
) -> None:
    hostname_per_cdn = score_config["hostname_per_cdn"]
    total_hostnames = set()
    for hostnames in hostname_per_cdn.values():
        total_hostnames.update(hostnames)

    logger.info(f"{len(total_hostnames)=}")
    logger.info(f"nb targets:: {len(results)}")

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
                ax1.plot(
                    x,
                    y,
                    label=f"ECS SP, {budget} VPs, {', ' + granularity if granularity else ''}",
                )

                logger.info(
                    f"ECS shortest ping:: {metric}, {budget} VPs, median_error={round(m_error, 2)} [km], {len(total_hostnames)} hostnames"
                )

    if metric_evaluated == "d_error":
        no_ping_per_metric = defaultdict(list)
        for target, target_results_per_metric in results.items():
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
            ax1.plot(
                x,
                y,
                label=f"ZP, {len(total_hostnames)} hostnames {', ' + granularity if granularity else ''}, {metric}",
            )
            m_error = round(
                np.median([r for r in cdf]),
                2,
            )

            logger.info(f"Zero ping:: {metric}, median_error={round(m_error, 2)} [km]")

    logger.info("###############################")


def tier1_plot(
    eval_dir: Path,
    granularities: list[str] = ["anser_bgp_prefixes"],
    score_metrics=["jaccard"],
    probing_budgets_evaluated: list[int] = [10, 20, 50],
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

        add_granularity_in_label = False
        if len(granularities) > 1:
            add_granularity_in_label = True

        if "answers" in granularities:
            logger.info("Answers granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answers,
                score_metrics=score_metrics,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answers" if add_granularity_in_label else "",
                metric_evaluated=metric_evaluated,
            )

        if "answer_subnets" in granularities:
            logger.info("Answer subnet granularity::")
            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answer_subnets,
                score_metrics=score_metrics,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answer subnets" if add_granularity_in_label else "",
                metric_evaluated=metric_evaluated,
            )

        if "answer_subnets" in granularities:
            logger.info("Answer BGP prefixes granularity::")
            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answer_bgp_prefixes,
                score_metrics=score_metrics,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answer bgp prefixes" if add_granularity_in_label else "",
                metric_evaluated=metric_evaluated,
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


def plot_all_results(
    eval_dir: Path,
    granularities: list[str] = ["anser_bgp_prefixes"],
    probing_budgets_evaluated: list[int] = [10, 20, 50],
    score_metrics=["jaccard"],
    metric_evaluated: str = "d_error",
    filter_org: list[str] = None,
    legend_outside: bool = False,
) -> None:
    fig, ax1 = plt.subplots(1, 1)

    ref_shortest_ping_results = load_pickle(
        path_settings.RESULTS_PATH / "results_ref_shortest_ping.pickle"
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

        add_granularity_in_label = False
        if len(granularities) > 1:
            add_granularity_in_label = True

        if "answers" in granularities:
            logger.info("Answers granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answers,
                score_metrics=score_metrics,
                metric_evaluated=metric_evaluated,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answers" if add_granularity_in_label else "",
            )

        if "answer_subnets" in granularities:
            logger.info("Answer subnets granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                metric_evaluated=metric_evaluated,
                score_metrics=score_metrics,
                results=eval.results_answer_subnets,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answer subnets" if add_granularity_in_label else "",
            )

        if "answer_bgp_prefixes" in granularities:
            logger.info("Answer BGP prefixes granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answer_bgp_prefixes,
                score_metrics=score_metrics,
                metric_evaluated=metric_evaluated,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answer bgp prefixes" if add_granularity_in_label else "",
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

    x_label = ""
    if metric_evaluated == "d_error":
        x_label = "geolocation error [km]"
    if metric_evaluated == "rtt":
        x_label = "shortest ping RTT [ms]"

    plt.xlabel(x_label)
    plt.ylabel("proportion of targets")
    plt.legend(loc="upper left", fontsize=8)
    if legend_outside:
        plt.legend(bbox_to_anchor=(1, 1))
    plt.xscale("log")
    plt.grid()
    plt.savefig(
        path_settings.FIGURE_PATH / f"tier3_{metric_evaluated}.png", bbox_inches="tight"
    )
    plt.show()


def plot_end_to_end_results(
    eval_dir: Path,
    granularities: list[str] = ["anser_bgp_prefixes"],
    probing_budgets_evaluated: list[int] = [10, 20, 50],
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

        add_granularity_in_label = False
        if len(granularities) > 1:
            add_granularity_in_label = True

        if "answer_subnets" in granularities:
            logger.info("Answer subnets granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                metric_evaluated=metric_evaluated,
                score_metrics=score_metrics,
                results=eval.results_answer_subnets,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answer subnets" if add_granularity_in_label else "",
            )

        if "answer_bgp_prefixes" in granularities:
            logger.info("Answer BGP prefixes granularity::")

            plot_per_granularity(
                ax1=ax1,
                score_config=score_config,
                results=eval.results_answer_bgp_prefixes,
                score_metrics=score_metrics,
                metric_evaluated=metric_evaluated,
                probing_budgets_evaluated=probing_budgets_evaluated,
                granularity="answer bgp prefixes" if add_granularity_in_label else "",
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


if __name__ == "__main__":
    plot_end_to_end_results(
        eval_dir=path_settings.RESULTS_PATH / "tier3_evaluation",
        granularities=["answer_subnets"],
        score_metrics=["jaccard"],
        probing_budgets_evaluated=[10, 20, 50],
        imc_nb_vps=[10],
        metric_evaluated="d_error",
    )
