import numpy as np
from collections import defaultdict
from loguru import logger
from pyasn import pyasn
from pathlib import Path

from geogiant.evaluation.hostname import (
    select_hostname_per_org_per_ns,
    get_all_name_servers,
)
from geogiant.common.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
    load_targets,
    load_vps,
)
from geogiant.common.utils import (
    get_parsed_vps,
    EvalResults,
    TargetScores,
)
from geogiant.evaluation.plot import (
    plot_ref,
    geo_resolver_cdfs,
    get_proportion_under,
    plot_multiple_cdf,
)
from geogiant.evaluation.ecs_geoloc_utils import ecs_dns_vp_selection_eval
from geogiant.evaluation.scores import get_scores
from geogiant.common.files_utils import load_csv, load_json, load_pickle, dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def load_hostnames() -> tuple[dict]:
    # select hostnames with: 1) only one large hosting organization, 2) at least two bgp prefixes
    best_hostnames_per_org_per_ns = load_json(
        path_settings.DATASET
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
        "TWITTER",
        "MICROSOFT",
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
    targets_table = clickhouse_settings.VPS_FILTERED_TABLE
    vps_table = clickhouse_settings.VPS_FILTERED_TABLE

    targets_ecs_table = "vps_ecs_mapping"
    vps_ecs_table = "vps_ecs_mapping"

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
        "hostname_/storage/hugo/geogiant/geogiant/tmpselection": "max_bgp_prefix",
        "score_metric": ["jaccard"],
        "answer_granularities": ["answer_subnets"],
        "output_path": output_path,
    }

    get_scores(score_config)


def compute_score() -> None:
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
    if not output_path.exists():
        score_per_config(all_orgs_hostnames, output_path)

    #############################################################################################################################
    logger.info(f"AKAMAI hostnames:: {len(akamai_hostnames)} orgs")
    output_path = path_settings.RESULTS_PATH / f"tier2_evaluation/scores__akamai.pickle"
    if not output_path.exists():
        score_per_config(akamai_hostnames, output_path)


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    probing_budgets = [5, 10, 20, 30, 50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.VPS_VPS_MESHED_PINGS_TABLE, removed_vps
    )
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

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


if __name__ == "__main__":
    compute_scores = False
    evaluation = False
    make_figure = True

    if compute_scores:
        compute_score()

    if evaluation:
        evaluate()
    if make_figure:
        plot()
