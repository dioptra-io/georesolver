import os

from collections import defaultdict
from loguru import logger
from pyasn import pyasn
from pathlib import Path
from pprint import pprint

from georesolver.clickhouse.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
    load_targets,
    load_vps,
)
from georesolver.common.utils import (
    get_parsed_vps,
    EvalResults,
    TargetScores,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_ref,
    geo_resolver_cdfs,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.evaluation.evaluation_ecs_geoloc_functions import (
    ecs_dns_vp_selection_eval,
)
from georesolver.evaluation.evaluation_score_functions import get_scores
from georesolver.common.files_utils import (
    load_json,
    dump_json,
    load_pickle,
    dump_pickle,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["CLICKHOUSE_DATABASE"] = ch_settings.CLICKHOUSE_DATABASE_EVAL


def load_hostnames() -> tuple[dict]:
    # select hostnames with: 1) only one large hosting organization, 2) at least two bgp prefixes
    best_hostnames_per_org_per_ns = load_json(
        path_settings.HOSTNAME_FILES
        / "hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json",
    )

    best_hostnames_per_org = defaultdict(list)
    for ns in best_hostnames_per_org_per_ns:
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            best_hostnames_per_org[org].extend(hostnames)

    hg_orgs = [
        "AMAZON",
        "GOOGLE",
        "FACEBOOK",
        "AKAMAI",
        "ALIBABA-CN-NET",
        "OVH",
        "CDNNETWORKS",
        "APPLE",
        "CDN77",
        "INCAPSULA",
        "FASTLY",
    ]

    hg_hostnames = defaultdict(list)
    no_hg_hostnames = defaultdict(list)
    all_orgs_hostnames = defaultdict(list)
    akamai_hostnames = defaultdict(list)
    for org, hostnames in best_hostnames_per_org.items():
        if org in hg_orgs:
            hg_hostnames[org].extend(hostnames)

        if org not in hg_orgs:
            no_hg_hostnames[org].extend(hostnames)

        all_orgs_hostnames[org].extend(hostnames)

        if org == "AKAMAI":
            akamai_hostnames["AKAMAI"].extend(hostnames)

    return hg_hostnames, no_hg_hostnames, all_orgs_hostnames, akamai_hostnames


def score_per_config(hostnames_per_org: dict[str], output_path: Path) -> None:
    targets_table = ch_settings.VPS_FILTERED_TABLE
    vps_table = ch_settings.VPS_FILTERED_TABLE
    targets_ecs_table = ch_settings.VPS_ECS_MAPPING_TABLE
    vps_ecs_table = ch_settings.VPS_ECS_MAPPING_TABLE

    selected_hostnames = set()
    for org, hostnames in hostnames_per_org.items():
        logger.debug(f"{org=}, {len(hostnames)=}")
        selected_hostnames.update(hostnames)

    logger.info(f"{len(selected_hostnames)=}")

    score_config = {
        "targets_table": targets_table,
        "main_org_threshold": 0.0,
        "bgp_prefixes_threshold": 0.0,
        "vps_table": vps_table,
        "hostname_per_cdn": hostnames_per_org,
        "targets_ecs_table": targets_ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_/storage/hugo/georesolver/georesolver/tmpselection": "max_bgp_prefix",
        "score_metric": ["jaccard"],
        "answer_granularities": ["answer_subnets"],
        "output_path": output_path,
    }

    get_scores(score_config)


def compute_score(ordered_hg: list) -> None:
    """calculate score for each organization/ns pair"""

    hg_hostnames, no_hg_hostnames, all_orgs_hostnames, akamai_hostnames = (
        load_hostnames()
    )

    ##############################################################################################################################
    logger.info(f"HG hostnames:: {len(hg_hostnames)} orgs")
    output_path = (
        path_settings.RESULTS_PATH / f"tier2_evaluation/scores__hg_orgs.pickle"
    )
    if not output_path.exists():
        score_per_config(hg_hostnames, output_path)

    ##############################################################################################################################
    logger.info(f"No HG hostnames:: {len(no_hg_hostnames)} orgs")
    output_path = (
        path_settings.RESULTS_PATH / f"tier2_evaluation/scores__no_hg_orgs.pickle"
    )
    if not output_path.exists():
        score_per_config(no_hg_hostnames, output_path)

    ##############################################################################################################################
    logger.info(f"All orgs hostnames:: {len(all_orgs_hostnames)} orgs")
    output_path = (
        path_settings.RESULTS_PATH / f"tier2_evaluation/scores__all_orgs.pickle"
    )
    # if not output_path.exists():
    #     score_per_config(all_orgs_hostnames, output_path)

    #############################################################################################################################
    hostname_configs = []
    # create a list of hostname selection (remove one new org each time)
    for i in range(len(ordered_hg) - 1):
        new_config_hostnames = {}
        # remove the n first hg
        for org, hostnames in all_orgs_hostnames.items():
            if org in ordered_hg[: (i + 1)]:
                continue
            new_config_hostnames[org] = hostnames

        # add config for score calculation
        hostname_configs.append(new_config_hostnames)
        logger.info(f"nb orgs for hostnames:: {len(new_config_hostnames.keys())}")

        # save config for checking
        removed_hg = "_".join([ordered_hg[j] for j in range(0, i + 1)])

        output_path = (
            path_settings.RESULTS_PATH
            / f"tier2_evaluation/hostnames__georesolver_minus_{removed_hg}.json"
        )
        dump_json(new_config_hostnames, output_path)

    for i, hostname_config in enumerate(hostname_configs):
        removed_hg = "_".join([ordered_hg[j] for j in range(0, i + 1)])

        logger.info(f"Georesolver minus {removed_hg}:: {len(hostname_config)} orgs")

        output_path = (
            path_settings.RESULTS_PATH
            / f"tier2_evaluation/scores__georesolver_minus_{removed_hg}.pickle"
        )

        if not output_path.exists():
            score_per_config(hostname_config, output_path)

    #############################################################################################################################
    logger.info(f"AKAMAI hostnames:: {len(akamai_hostnames)} orgs")
    output_path = path_settings.RESULTS_PATH / f"tier2_evaluation/scores__akamai.pickle"
    if not output_path.exists():
        score_per_config(akamai_hostnames, output_path)


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    probing_budgets = [50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    targets = load_targets(ch_settings.VPS_FILTERED_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(
        path_settings.DATASET / "imc2024_generated_files/removed_vps.json"
    )
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    ping_vps_to_target = get_pings_per_target(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    logger.info("BGP prefix score geoloc evaluation")

    score_dir = path_settings.RESULTS_PATH / "tier2_evaluation"

    for score_file in score_dir.iterdir():

        if "results" in score_file.name:
            continue

        scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)

        logger.info(f"ECS evaluation for score:: {score_file}")

        output_file = (
            path_settings.RESULTS_PATH
            / f"tier2_evaluation/{'results' + str(score_file).split('scores')[-1]}"
        )

        if output_file.exists():
            continue

        results_answer_subnets = ecs_dns_vp_selection_eval(
            targets=targets,
            vps_per_subnet=vps_per_subnet,
            subnet_scores=scores.score_answer_subnets,
            ping_vps_to_target=ping_vps_to_target,
            last_mile_delay=last_mile_delay,
            vps_coordinates=vps_coordinates,
            probing_budgets=probing_budgets,
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


def plot(
    output_path=path_settings.FIGURE_PATH / "hg_vs_no_hg_vs_all.png",
    metric_evaluated="d_error",
    legend_pos="lower right",
) -> None:
    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval_files = [
        "results__all_orgs.pickle",
        "results__hg_orgs.pickle",
        "results__akamai.pickle",
    ]

    labels = ["All (GeoResolver)", "Hypergiants", "Akamai"]
    for i, file in enumerate(eval_files):

        eval: EvalResults = load_pickle(
            path_settings.RESULTS_PATH / f"tier2_evaluation/{file}"
        )

        score_config = eval.target_scores.score_config
        selected_hostnames_per_cdn = score_config["hostname_per_cdn"]
        nb_orgs = len(selected_hostnames_per_cdn)

        selected_hostnames = set()
        for org, hostnames in selected_hostnames_per_cdn.items():
            selected_hostnames.update(hostnames)

        nb_hostnames = len(selected_hostnames)

        label = labels[i]

        logger.info(f"{file=} loaded")
        logger.info(f"{nb_orgs} orgs, {nb_hostnames} hostnames")

        cdfs = geo_resolver_cdfs(
            results=eval.results_answer_subnets,
            metric_evaluated=metric_evaluated,
            label=label,
        )
        all_cdfs.extend(cdfs)

    plot_multiple_cdf(all_cdfs, output_path, metric_evaluated, False, legend_pos)


def order_hg() -> list:
    """order HG based on the fraction of IP addresses geolocated under 40km"""
    results_path = path_settings.RESULTS_PATH / "tier1_evaluation"

    frac_under_40_per_hg = {}

    for i, file in enumerate(results_path.iterdir()):
        if "score" in file.name:
            continue

        hg = file.name.split("_")[-1].split(".")[0]

        logger.info(f"Analysing results for {hg}")

        eval: EvalResults = load_pickle(file)
        d_errors_per_budget = defaultdict(list)
        for _, target_results in eval.results_answer_subnets.items():
            try:
                results = target_results["result_per_metric"]["jaccard"]
                shortest_ping_vp_per_budget: dict = results[
                    "ecs_shortest_ping_vp_per_budget"
                ]
            except KeyError:
                continue

            for budget, shortest_ping_vp in shortest_ping_vp_per_budget.items():
                d_errors_per_budget[budget].append(shortest_ping_vp["d_error"])

        # get cdf for each budget/rank
        cdfs = []
        for budget, d_errors in d_errors_per_budget.items():
            x, y = ecdf(d_errors)
            cdfs.append((x, y, f"{hg}"))

            frac_under_40km = round(get_proportion_under(x, y, 40), 2)
            frac_under_2ms = round(get_proportion_under(x, y, 2), 2)

            logger.info(f"{hg} :: {frac_under_40km=}")
            logger.info(f"{hg} :: {frac_under_2ms=}")

            frac_under_40_per_hg[hg] = frac_under_40km

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="tire1_hg_ordering",
        metric_evaluated="d_error",
    )

    frac_under_40_per_hg = sorted(
        frac_under_40_per_hg.items(), key=lambda x: x[-1], reverse=True
    )

    logger.info(f"{frac_under_40_per_hg=}")

    return frac_under_40_per_hg


if __name__ == "__main__":
    compute_scores = True
    evaluation = True
    make_figure = True

    if compute_scores:
        # frac_under_40_per_hg = order_hg()
        frac_under_40_per_hg = [
            ("INCAPSULA", 0.6),
            ("APPLE", 0.59),
            ("FACEBOOK", 0.55),
            ("GOOGLE", 0.46),
            ("AMAZON", 0.43),
            ("AKAMAI", 0.38),
            ("CDN77", 0.36),
            ("ALIBABA-CN-NET", 0.34),
            ("OVH", 0.23),
            ("FASTLY", 0.19),
        ]
        ordered_hg = [hg for hg, _ in frac_under_40_per_hg]
        compute_score(ordered_hg)

    if evaluation:
        evaluate()
    if make_figure:
        plot()
