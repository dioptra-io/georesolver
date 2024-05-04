"""methods for plotting graph"""

import numpy as np

from pathlib import Path
from loguru import logger
from collections import defaultdict, OrderedDict

from geogiant.evaluation.plot import ecdf
from geogiant.common.files_utils import load_pickle
from geogiant.common.utils import EvalResults
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


def get_proportion_under(x, y, threshold: int = 40) -> int:
    for i, distance in enumerate(x):
        if distance > threshold:
            proportion_of_ip = y[i]
            break

    return round(proportion_of_ip, 2)


def zero_ping_data(
    results: dict,
    score_metrics: list[str] = ["jaccard"],
    metric_evaluated: str = "d_error",
    label_granularity: str = "",
    label_hostnames: int = None,
    label_orgs: int = None,
    label_bgp_prefix: int = None,
    label_nb_hostname_per_org_ns: int = None,
) -> list[tuple]:
    data = []
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

        m_error = round(np.median(x), 2)
        proportion_of_ip = get_proportion_under(x, y)

        data.append(
            (
                "ZP",
                label_bgp_prefix,
                label_nb_hostname_per_org_ns,
                label_hostnames,
                m_error,
                proportion_of_ip,
            )
        )

        logger.info(f"Zero ping:: {metric}, median_error={round(m_error, 2)} [km]")

    return data


def get_data_from_results(
    results: dict,
    probing_budgets_evaluated: list[int] = [50],
    score_metrics: list[str] = ["jaccard"],
    metric_evaluated: str = "d_error",
    label_granularity: str = "",
    label_orgs: int = None,
    label_hostnames: int = None,
    label_bgp_prefix: int = None,
    label_nb_hostname_per_org_ns: int = None,
) -> list[tuple]:
    data = []
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
                if label_nb_hostname_per_org_ns:
                    label += f", {label_nb_hostname_per_org_ns} hostnames per NS/org"

                proportion_of_ip = get_proportion_under(x, y)

                data.append(
                    (
                        "SP",
                        label_bgp_prefix,
                        label_nb_hostname_per_org_ns,
                        label_hostnames,
                        m_error,
                        proportion_of_ip,
                    )
                )
                logger.info(
                    f"ECS SP:: {metric}, <40km={proportion_of_ip}, {label_hostnames} hostnames"
                )
                logger.info(
                    f"ECS SP:: {metric}, median_error={m_error} [km], {label_hostnames} hostnames"
                )

    zp_data = zero_ping_data(
        results=results,
        score_metrics=score_metrics,
        metric_evaluated=metric_evaluated,
        label_granularity=label_granularity,
        label_hostnames=label_hostnames,
        label_orgs=label_orgs,
        label_bgp_prefix=label_bgp_prefix,
        label_nb_hostname_per_org_ns=label_nb_hostname_per_org_ns,
    )
    data.extend(zp_data)

    return data


def bgp_prefix_threshold_data(
    eval_dir: Path,
    metric_evaluated: str = "d_error",
) -> None:

    results_files = defaultdict(list)
    for file in eval_dir.iterdir():
        if "result" in file.name and "hostnames_per_org_ns" in file.name:
            bgp_prefix_threshold = file.name.split("score_")[-1].split("_")[0]
            nb_hostnames_per_org_ns = file.name.split("BGP_")[-1].split("_")[0]
            results_files[int(bgp_prefix_threshold)].append(
                (int(nb_hostnames_per_org_ns), file)
            )

    for bgp_prefix_threshold in results_files:
        results_files[bgp_prefix_threshold] = sorted(
            results_files[bgp_prefix_threshold], key=lambda x: x[0]
        )
    results_files = OrderedDict(sorted(results_files.items(), key=lambda x: x[0]))

    all_data = []
    for bgp_prefix_threshold in results_files:
        for nb_hostnames_per_org_ns, file in results_files[bgp_prefix_threshold]:

            if bgp_prefix_threshold in [5, 10, 20, 50, 100]:
                if nb_hostnames_per_org_ns in [3, 5, 10]:

                    logger.info(f"{bgp_prefix_threshold=}")

                    eval: EvalResults = load_pickle(file)

                    logger.info(f"{file=} loaded")
                    score_config = eval.target_scores.score_config
                    hostname_per_cdn = score_config["hostname_per_cdn"]

                    total_hostnames = set()
                    for _, hostnames in hostname_per_cdn.items():
                        total_hostnames.update(hostnames)

                    data = get_data_from_results(
                        results=eval.results_answer_subnets,
                        score_metrics=["jaccard"],
                        metric_evaluated=metric_evaluated,
                        label_hostnames=len(total_hostnames),
                        label_bgp_prefix=bgp_prefix_threshold,
                        label_nb_hostname_per_org_ns=nb_hostnames_per_org_ns,
                    )
                    all_data.extend(data)

    return all_data
