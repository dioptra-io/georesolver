"""EVALUATION:: """

from collections import defaultdict
from loguru import logger
from pyasn import pyasn

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
from geogiant.evaluation.ecs_geoloc_utils import ecs_dns_vp_selection_eval
from geogiant.evaluation.scores import get_scores
from geogiant.common.files_utils import load_csv, load_json, load_pickle, dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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

    selected_hostnames = select_hostname_per_org_per_ns(
        name_servers_per_hostname,
        tlds,
        cdn_per_hostname,
        bgp_prefix_per_hostname,
        main_org_threshold,
        bgp_prefixes_threshold,
    )

    return selected_hostnames


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    targets_table = clickhouse_settings.VPS_FILTERED_TABLE
    vps_table = clickhouse_settings.VPS_FILTERED_TABLE

    targets_ecs_table = "vps_ecs_mapping"
    vps_ecs_table = "vps_ecs_mapping"

    main_org_threshold = 0.8
    bgp_prefixes_threshold = 2

    # select hostnames with: 1) only one large hosting organization, 2) at least two bgp prefixes
    hostname_per_ns_per_org = load_json(
        path_settings.DATASET / "best_hostnames_per_org_per_ns.json"
    )

    org_per_ns = {
        "awsdns": "AMAZON",
        "google": "GOOGLE",
        "facebook": "FACEBOOK",
        "akamai": "AKAMAI",
        "impervadns": "INCAPSULA",
        "bunny": "CDN77",
        "dns-parking": "AS-HOSTINGER",
    }

    nb_hostnames_per_org = [1, 10, 100, 500]
    for nb_hostnames in nb_hostnames_per_org:
        logger.info(f"{nb_hostnames=}")
        for evaluated_ns, evaluated_org in org_per_ns.items():

            # extract hostname per cdn
            selected_hostnames_per_cdn = defaultdict(list)
            for ns in hostname_per_ns_per_org:
                if ns != evaluated_ns:
                    continue

                for org, hostnames in hostname_per_ns_per_org[ns].items():
                    if org != evaluated_org:
                        continue

                    selected_hostnames_per_cdn[org].extend(hostnames[:nb_hostnames])

            selected_hostnames = set()
            for org, hostnames in selected_hostnames_per_cdn.items():
                selected_hostnames.update(hostnames)

            output_path = (
                path_settings.RESULTS_PATH
                / f"tier1_evaluation/scores__{len(selected_hostnames)}_hostname_{evaluated_org}_ns_{evaluated_ns}.pickle"
            )

            # some organizations do not have enought hostnames
            if output_path.exists():
                continue

            for org, hostnames in selected_hostnames_per_cdn.items():
                logger.info(f"{org=}, {len(hostnames)=}")

            score_config = {
                "targets_table": targets_table,
                "main_org_threshold": main_org_threshold,
                "bgp_prefixes_threshold": bgp_prefixes_threshold,
                "vps_table": vps_table,
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
    probing_budgets = [5, 10, 20, 30, 50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    logger.info("BGP prefix score geoloc evaluation")

    score_dir = path_settings.RESULTS_PATH / "tier1_evaluation"

    for score_file in score_dir.iterdir():

        if "results" in score_file.name:
            continue

        scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)

        logger.info(f"ECS evaluation for score:: {score_file}")

        output_file = (
            path_settings.RESULTS_PATH
            / f"tier1_evaluation/{'results' + str(score_file).split('scores')[-1]}"
        )

        if output_file.exists():
            continue

        results_answers = {}
        results_answer_subnets = {}
        results_answer_bgp_prefixes = {}
        if "answers" in scores.score_config["answer_granularities"]:
            results_answers = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        if "answer_subnets" in scores.score_config["answer_granularities"]:
            results_answer_subnets = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        if "answer_bgp_prefixes" in scores.score_config["answer_granularities"]:
            results_answer_bgp_prefixes = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_bgp_prefixes,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        results = EvalResults(
            target_scores=scores,
            results_answers=results_answers,
            results_answer_subnets=results_answer_subnets,
            results_answer_bgp_prefixes=results_answer_bgp_prefixes,
        )

        logger.info(f"output file:: {output_file}")

        dump_pickle(
            data=results,
            output_file=output_file,
        )


if __name__ == "__main__":
    compute_scores = False
    evaluation = False

    if compute_scores:
        compute_score()

    if evaluation:
        evaluate()
