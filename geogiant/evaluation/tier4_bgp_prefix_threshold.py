import os
import numpy as np

from pathlib import Path
from collections import defaultdict, OrderedDict
from loguru import logger
from pyasn import pyasn

from geogiant.evaluation.evaluation_hostname_functions import (
    select_hostname_per_org_per_ns,
    get_all_name_servers,
    parse_name_servers,
    get_hostname_per_name_server,
)
from geogiant.clickhouse.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
    load_targets,
    load_vps,
)
from geogiant.common.utils import (
    get_parsed_vps,
    get_vps_country,
    EvalResults,
    TargetScores,
)
from geogiant.evaluation.evaluation_ecs_geoloc_functions import ecs_dns_vp_selection_eval
from geogiant.evaluation.evaluation_score_functions import get_scores
from geogiant.evaluation.evaluation_plot_functions import (
    plot_multiple_cdf,
    plot_ecs_shortest_ping,
    plot_ref,
    ecdf,
    get_proportion_under,
)
from geogiant.common.files_utils import load_csv, load_json, load_pickle, dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["CLICKHOUSE_DATABASE"] = ch_settings.CLICKHOUSE_DATABASE_EVAL


def get_bgp_prefixes_per_hostname(cdn_per_hostname: dict) -> dict:
    """return the number of unique bgp prefixes per hostname"""
    bgp_prefix_per_hostname = defaultdict(set)
    for hostname, bgp_prefixes_per_cdn in cdn_per_hostname.items():
        for bgp_prefixes in bgp_prefixes_per_cdn.values():
            bgp_prefix_per_hostname[hostname].update(bgp_prefixes)

    return bgp_prefix_per_hostname


def select_hostnames(
    cdn_per_hostname: dict,
    bgp_prefix_per_hostname: dict,
    main_org_threshold: float,
    bgp_prefixes_threshold: int,
) -> dict:
    """select hostnames, order them per name server and hosting organization, filter function of parameters"""
    name_servers_per_hostname = get_all_name_servers()
    tlds = load_csv(path_settings.DATASET / "tlds.csv")
    tlds = [t.lower() for t in tlds]

    name_servers_per_hostname = parse_name_servers(name_servers_per_hostname, tlds)

    hostname_per_name_servers = get_hostname_per_name_server(name_servers_per_hostname)

    selected_hostnames = select_hostname_per_org_per_ns(
        name_servers_per_hostname,
        tlds,
        cdn_per_hostname,
        bgp_prefix_per_hostname,
        main_org_threshold,
        bgp_prefixes_threshold,
    )

    return selected_hostnames


def get_ns_per_hostname() -> dict:
    name_servers_per_hostname = get_all_name_servers()
    tlds = load_csv(path_settings.DATASET / "tlds.csv")
    tlds = [t.lower() for t in tlds]

    name_servers_per_hostname = parse_name_servers(name_servers_per_hostname, tlds)
    hostname_per_name_servers = get_hostname_per_name_server(name_servers_per_hostname)

    ns_per_hostname = {}
    for ns, hostnames in hostname_per_name_servers:
        for hostname in hostnames:
            ns_per_hostname[hostname] = ns

    return ns_per_hostname


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    targets_table = ch_settings.VPS_FILTERED_TABLE
    vps_table = ch_settings.VPS_FILTERED_TABLE

    targets_ecs_table = "vps_ecs_mapping"
    vps_ecs_table = "vps_ecs_mapping"

    for bgp_threshold in [5, 10, 20, 50, 100]:
        for nb_hostname_per_ns_org in [3, 5, 10]:

            selected_hostnames_per_cdn_per_ns = load_json(
                path_settings.DATASET
                / f"hostname_geo_score_selection_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.json"
            )

            selected_hostnames = set()
            selected_hostnames_per_cdn = defaultdict(list)
            for ns in selected_hostnames_per_cdn_per_ns:
                for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
                    selected_hostnames.update(hostnames)
                    selected_hostnames_per_cdn[org].extend(hostnames)

            logger.info(f"{len(selected_hostnames)=}")

            output_path = (
                path_settings.RESULTS_PATH
                / f"tier4_evaluation/scores__best_hostname_geo_score_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.pickle"
            )

            # some organizations do not have enought hostnames
            if output_path.exists():
                logger.info(
                    f"Score for {bgp_threshold} BGP prefix threshold alredy done"
                )
                continue

            score_config = {
                "targets_table": targets_table,
                "main_org_threshold": 0.0,
                "bgp_prefixes_threshold": 0.0,
                "vps_table": vps_table,
                "hostname_per_cdn_per_ns": selected_hostnames_per_cdn_per_ns,
                "hostname_per_cdn": selected_hostnames_per_cdn,
                "targets_ecs_table": targets_ecs_table,
                "vps_ecs_table": vps_ecs_table,
                "hostname_selection": "max_bgp_prefix",
                "score_metric": ["jaccard"],
                "answer_granularities": ["answer_subnets"],
                "output_path": output_path,
            }

            get_scores(score_config)


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    probing_budgets = [50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        ch_settings.VPS_MESHED_TRACEROUTE_TABLE
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        ch_settings.VPS_VPS_MESHED_PINGS_TABLE, removed_vps
    )
    targets = load_targets(ch_settings.VPS_FILTERED_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    vps_country = get_vps_country(vps)

    logger.info("BGP prefix score geoloc evaluation")

    score_dir = path_settings.RESULTS_PATH / "tier4_evaluation"

    for score_file in score_dir.iterdir():

        if "results" in score_file.name:
            continue

        scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)

        logger.info(f"ECS evaluation for score:: {score_file}")

        output_file = (
            path_settings.RESULTS_PATH
            / f"tier4_evaluation/{'results' + str(score_file).split('scores')[-1]}"
        )

        if "20_BGP_3" in score_file.name:
            # if output_file.exists():
            #     continue

            results_answer_subnets = {}
            results_answer_subnets = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
                vps_country=vps_country,
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


def plot_bgp_prefix_threshold(
    eval_dir: Path,
    output_path: Path,
    metric_evaluated: str = "d_error",
    plot_zp: bool = False,
    legend_outside: bool = False,
    legend_pos: str = "lower right",
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

    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)
    for bgp_prefix_threshold in results_files:
        for nb_hostnames_per_org_ns, file in results_files[bgp_prefix_threshold]:

            if bgp_prefix_threshold in [20, 100]:
                if nb_hostnames_per_org_ns in [3]:

                    # if bgp_prefix_threshold in [5, 10, 20, 50, 100]:
                    #     if nb_hostnames_per_org_ns in [3, 5, 10]:

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
                        label_nb_hostname_per_org_ns=nb_hostnames_per_org_ns,
                        plot_zp=plot_zp,
                    )
                    all_cdfs.extend(cdfs)

    plot_multiple_cdf(
        all_cdfs, output_path, metric_evaluated, legend_outside, legend_pos=legend_pos
    )


def zero_pinp_major_country(
    results: dict,
    score_metrics: list[str] = ["jaccard"],
) -> list[tuple]:
    no_ping_per_metric = defaultdict(list)
    for _, target_results_per_metric in results.items():
        for metric, target_results in target_results_per_metric[
            "result_per_metric"
        ].items():
            if not metric in score_metrics:
                continue

            target_country = target_results["no_ping_vp"]["country"]
            major_vp_country, proportion = target_results["no_ping_vp"]["major_country"]

            no_ping_per_metric[metric].append(
                (target_country, major_vp_country, proportion)
            )

    correct_country = 0
    proportion_threhsold = 0.8
    correct_country_fct_proportion = []
    for metric, major_countries in no_ping_per_metric.items():
        for target_country, major_vp_country, proportion in major_countries:
            if target_country == major_vp_country:
                correct_country += 1

            if proportion > proportion_threhsold:
                correct_country_fct_proportion.append(
                    1 if target_country == major_vp_country else 0
                )

        proportion = correct_country * 100 / len(major_countries)
        logger.info(f"Zero ping:: geolocation country level={round(proportion,2)} [%]")
        proportion = (
            correct_country_fct_proportion.count(1)
            * 100
            / len(correct_country_fct_proportion)
        )
        logger.info(
            f"Zero ping:: geolocation country level={round(proportion,2)} ({len(correct_country_fct_proportion)}) [%]"
        )


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


def plot_bgp_prefix_threshold_data(
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


if __name__ == "__main__":
    compute_scores = False
    evaluation = False
    plot = True

    if compute_scores:
        compute_score()

    if evaluation:
        evaluate()

    if plot:
        plot_bgp_prefix_threshold(
            eval_dir=path_settings.RESULTS_PATH / "tier4_evaluation",
            output_path="bgp_prefix_threshold",
            metric_evaluated="d_error",
            plot_zp=False,
        )

        plot_bgp_prefix_threshold_data(
            eval_dir=path_settings.RESULTS_PATH / "tier4_evaluation",
            metric_evaluated="d_error",
        )
