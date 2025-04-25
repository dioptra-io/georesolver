import os
import numpy as np

from collections import defaultdict
from loguru import logger
from georesolver.clickhouse.queries import (
    get_pings_per_target_extended,
    load_targets,
    load_vps,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_vp_selection_per_target,
)
from georesolver.common.files_utils import load_json, dump_json, load_pickle
from georesolver.common.utils import get_d_errors_georesolver, get_d_errors_ref
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

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

TARGETS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
TARGETS_ECS_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE
VPS_ECS_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE
RESULTS_PATH = path_settings.RESULTS_PATH / "hg_dependency"


def load_hostnames() -> tuple[dict, dict, dict, dict]:
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
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)

    (
        no_hio_hg_hostnames,
        no_nso_hg_hostnames,
        no_nso_no_hio_hg_hostnames,
        georesolver_hostnames,
    ) = load_hostnames()

    logger.info(f"Calculation score for no HIO")
    output_path = RESULTS_PATH / "scores__no_hio_hg_orgs.pickle"
    hostnames = set()
    for h in no_hio_hg_hostnames.values():
        hostnames.update(h)
    get_scores(
        output_path=output_path,
        hostnames=hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )

    logger.info(f"Calculation score for no NSO")
    output_path = RESULTS_PATH / "scores__no_nso_hg_orgs.pickle"
    hostnames = set()
    for h in no_nso_hg_hostnames.values():
        hostnames.update(h)
    get_scores(
        output_path=output_path,
        hostnames=hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )

    logger.info(f"Calculation score for no HIO || NSO")
    output_path = RESULTS_PATH / "scores__no_nso_no_hio_hg_orgs.pickle"
    hostnames = set()
    for h in no_nso_no_hio_hg_hostnames.values():
        hostnames.update(h)
    get_scores(
        output_path=output_path,
        hostnames=hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )

    logger.info(f"Calculation score for GeoResolver")
    output_path = RESULTS_PATH / "scores__georesolver.pickle"
    hostnames = set()
    for h in georesolver_hostnames.values():
        hostnames.update(h)
    get_scores(
        output_path=output_path,
        hostnames=hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )


def evaluate(only_hgs: bool = False) -> None:
    """perform vp selection based on targets/vps ecs redirection similarities"""

    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)

    for score_file in RESULTS_PATH.iterdir():

        if "scores" not in score_file.name:
            continue

        output_file = (
            RESULTS_PATH / f"{'results' + str(score_file).split('scores')[-1]}"
        )

        scores = load_pickle(score_file)

        get_vp_selection_per_target(
            output_path=output_file,
            scores=scores,
            targets=[t["addr"] for t in targets],
            vps=vps,
        )


def plot() -> None:
    cdfs = []
    eval_files = {
        "results__no_hio_hg_orgs.pickle": "No hypergiant HIO",
        "results__no_nso_hg_orgs.pickle": "No hypergiant NSO",
        "results__no_nso_no_hio_hg_orgs.pickle": "No hypergiant NSO/HIO",
        "results__georesolver.pickle": "GeoResolver",
    }

    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    # add reference
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs:: median_error={round(m_error, 2)} [km]")

    for file, label in eval_files.items():

        logger.info(f"GeoResolver from vp selection file:: {file}")

        vp_selection_per_target = load_pickle(RESULTS_PATH / file)

        d_errors = get_d_errors_georesolver(
            pings_per_target=pings_per_target,
            vp_selection_per_target=vp_selection_per_target,
            vps_coordinates=vps_coordinates,
        )

        # plot georesolver results
        x, y = ecdf(d_errors)
        cdfs.append((x, y, label))

        m_error = round(np.median(x), 2)
        proportion_of_ip = get_proportion_under(x, y)

        logger.info(f"GeoResolver:: {label}: <40km={round(proportion_of_ip, 2)}")
        logger.info(f"GeoResolver:: {label}: median_error={round(m_error, 2)} [km]")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="hg_dependency",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


if __name__ == "__main__":
    do_load_hostnames: bool = False
    do_score: bool = False
    do_eval: bool = False
    do_plot: bool = True

    if do_load_hostnames:
        load_hostnames()
    if do_score:
        compute_score()

    if do_eval:
        evaluate()

    if do_plot:
        plot()
