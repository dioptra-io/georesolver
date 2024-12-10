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

HG_ORGS = [
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

HG_NSO = [
    "awsdns",
    "google",
    "googledomains",
    "facebook",
    "instagram",
    "whatsapp",
    "akamai",
    "alibabadns",
    "aaplimg",
    "cdn77",
    "impervadns",
]

RESULTS_PATH = path_settings.RESULTS_PATH / "hg_dependency"


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


def load_hostnames() -> tuple[dict]:
    best_hostnames_per_org_per_ns = load_json(
        path_settings.HOSTNAME_FILES
        / "hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json",
    )

    # select hostnames that are not hosted by an HG
    selected_ns = set()
    selected_orgs = set()
    selected_hostnames = set()
    ns_org_pairs = set()
    no_hio_hg_hostnames = defaultdict(list)
    for ns in best_hostnames_per_org_per_ns:
        selected_ns.add(ns)
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            if org in HG_ORGS:
                continue
            selected_orgs.add(org)
            selected_hostnames.update(hostnames)
            no_hio_hg_hostnames[org].extend(hostnames)
            ns_org_pairs.add((ns, org))

    dump_json(no_hio_hg_hostnames, RESULTS_PATH / "no_hio_hg_hostnames.json")

    logger.info("No HIO from hypergiants")
    logger.info(f"Nb NSO           :: {len(selected_ns)}")
    logger.info(f"Nb HIO           :: {len(selected_orgs)}")
    logger.info(f"Nb Hostnames     :: {len(selected_hostnames)}")
    logger.info(f"Nb NSO/HIO pairs :: {len(ns_org_pairs)}")

    # select hostnames that is not served by an HG NS
    selected_ns = set()
    selected_orgs = set()
    selected_hostnames = set()
    ns_org_pairs = set()
    no_nso_hg_hostnames = defaultdict(list)
    for ns in best_hostnames_per_org_per_ns:
        if ns in HG_NSO:
            continue
        selected_ns.add(ns)
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            selected_orgs.add(org)
            selected_hostnames.update(hostnames)
            no_nso_hg_hostnames[org].extend(hostnames)
            ns_org_pairs.add((ns, org))

    logger.info("No NSO from hypergiants")
    logger.info(f"Nb NSO           :: {len(selected_ns)}")
    logger.info(f"Nb HIO           :: {len(selected_orgs)}")
    logger.info(f"Nb Hostnames     :: {len(selected_hostnames)}")
    logger.info(f"Nb NSO/HIO pairs :: {len(ns_org_pairs)}")

    dump_json(no_nso_hg_hostnames, RESULTS_PATH / "no_nso_hg_hostnames.json")

    # select hostnames are not hosted nor served by an HG
    selected_ns = set()
    selected_orgs = set()
    selected_hostnames = set()
    ns_org_pairs = set()
    no_nso_no_hio_hg_hostnames = defaultdict(list)
    for ns in best_hostnames_per_org_per_ns:
        if ns in HG_NSO:
            continue
        selected_ns.add(ns)
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            if org in HG_ORGS:
                continue
            selected_orgs.add(org)
            selected_hostnames.update(hostnames)
            no_nso_no_hio_hg_hostnames[org].extend(hostnames)
            ns_org_pairs.add((ns, org))

    logger.info("No HIO from hypergiants")
    logger.info(f"Nb NSO           :: {len(selected_ns)}")
    logger.info(f"Nb HIO           :: {len(selected_orgs)}")
    logger.info(f"Nb Hostnames     :: {len(selected_hostnames)}")
    logger.info(f"Nb NSO/HIO pairs :: {len(ns_org_pairs)}")

    dump_json(
        no_nso_no_hio_hg_hostnames, RESULTS_PATH / "no_nso_no_hio_hg_hostnames.json"
    )

    # selected Georesolver hostnames
    selected_ns = set()
    selected_orgs = set()
    selected_hostnames = set()
    ns_org_pairs = set()
    georesolver_hostnames = defaultdict(list)
    for ns in best_hostnames_per_org_per_ns:
        selected_ns.add(ns)
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            selected_orgs.add(org)
            selected_hostnames.update(hostnames)
            georesolver_hostnames[org].extend(hostnames)
            ns_org_pairs.add((ns, org))

    logger.info("Georesolver selection")
    logger.info(f"Nb NSO           :: {len(selected_ns)}")
    logger.info(f"Nb HIO           :: {len(selected_orgs)}")
    logger.info(f"Nb Hostnames     :: {len(selected_hostnames)}")
    logger.info(f"Nb NSO/HIO pairs :: {len(ns_org_pairs)}")

    dump_json(georesolver_hostnames, RESULTS_PATH / "georeoslver_hostnames.json")

    return (
        no_hio_hg_hostnames,
        no_nso_hg_hostnames,
        no_nso_no_hio_hg_hostnames,
        georesolver_hostnames,
    )


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    (
        no_hio_hg_hostnames,
        no_nso_hg_hostnames,
        no_nso_no_hio_hg_hostnames,
        georesolver_hostnames,
    ) = load_hostnames()

    logger.info(f"Calculation score for no HIO")
    output_path = RESULTS_PATH / "scores__no_hio_hg_orgs.pickle"
    if not output_path.exists():
        logger.info("skipped")
        score_per_config(no_hio_hg_hostnames, output_path)

    logger.info(f"Calculation score for no NSO")
    output_path = RESULTS_PATH / "scores__no_nso_hg_orgs.pickle"
    if not output_path.exists():
        logger.info("skipped")
        score_per_config(no_nso_hg_hostnames, output_path)

    logger.info(f"Calculation score for no HIO || NSO")
    output_path = RESULTS_PATH / "scores__no_nso_no_hio_hg_orgs.pickle"
    if not output_path.exists():
        logger.info("skipped")
        score_per_config(no_nso_no_hio_hg_hostnames, output_path)

    logger.info(f"Calculation score for GeoResolver")
    output_path = RESULTS_PATH / "scores__georesolver.pickle"
    if not output_path.exists():
        logger.info("skipped")
        score_per_config(georesolver_hostnames, output_path)


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    removed_vps = load_json(
        path_settings.DATASET / "imc2024_generated_files/removed_vps.json"
    )
    targets = load_targets(ch_settings.VPS_FILTERED_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    ping_vps_to_target = get_pings_per_target(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    for score_file in RESULTS_PATH.iterdir():

        if "scores" not in score_file.name:
            continue

        output_file = (
            RESULTS_PATH / f"{'results' + str(score_file).split('scores')[-1]}"
        )

        if output_file.exists():
            continue

        logger.info(f"ECS evaluation for score:: {score_file}")
        scores: TargetScores = load_pickle(RESULTS_PATH / score_file)

        results_answer_subnets = ecs_dns_vp_selection_eval(
            targets=targets,
            vps_per_subnet=vps_per_subnet,
            subnet_scores=scores.score_answer_subnets,
            ping_vps_to_target=ping_vps_to_target,
            last_mile_delay=last_mile_delay,
            vps_coordinates=vps_coordinates,
            probing_budgets=[50],
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
    output_path=path_settings.FIGURE_PATH / "hg_dependency.png",
    metric_evaluated="d_error",
    legend_pos="lower right",
) -> None:
    all_cdfs = []
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    eval_files = {
        "results__no_hio_hg_orgs.pickle": "No hypergiant HIO",
        "results__no_nso_hg_orgs.pickle": "No hypergiant NSO",
        "results__no_nso_no_hio_hg_orgs.pickle": "No hypergiant NSO/HIO",
        "results__georesolver.pickle": "GeoResolver",
    }

    for file, label in eval_files.items():

        eval: EvalResults = load_pickle(RESULTS_PATH / f"{file}")

        score_config = eval.target_scores.score_config
        selected_hostnames_per_cdn = score_config["hostname_per_cdn"]
        nb_orgs = len(selected_hostnames_per_cdn)

        selected_hostnames = set()
        for _, hostnames in selected_hostnames_per_cdn.items():
            selected_hostnames.update(hostnames)

        nb_hostnames = len(selected_hostnames)

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
    do_load_hostnames: bool = True
    do_score: bool = False
    do_eval: bool = False
    do_plot: bool = False

    if do_load_hostnames:
        load_hostnames()

    if do_score:
        compute_score()

    if do_eval:
        evaluate()
    if do_plot:
        plot()
