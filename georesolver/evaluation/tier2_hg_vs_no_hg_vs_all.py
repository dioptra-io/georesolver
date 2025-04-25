import os
import numpy as np

from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import (
    load_vps,
    load_targets,
    get_pings_per_target_extended,
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
from georesolver.common.files_utils import load_json, load_pickle
from georesolver.common.utils import get_d_errors_georesolver, get_d_errors_ref
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["CLICKHOUSE_DATABASE"] = ch_settings.CLICKHOUSE_DATABASE_EVAL

TARGES_TABLE = ch_settings.VPS_FILTERED_TABLE
VPS_TABLE = ch_settings.VPS_FILTERED_TABLE
TARGETS_ECS_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE
VPS_ECS_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE
RESULTS_PATH = path_settings.RESULTS_PATH / "tier2_evaluation"

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


def load_hostnames() -> tuple[dict, dict, dict, dict]:
    # select hostnames with: 1) only one large hosting organization, 2) at least two bgp prefixes
    best_hostnames_per_org_per_ns = load_json(
        path_settings.HOSTNAME_FILES
        / "hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json",
    )

    best_hostnames_per_org = defaultdict(list)
    for ns in best_hostnames_per_org_per_ns:
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            best_hostnames_per_org[org].extend(hostnames)

    hg_hostnames = defaultdict(list)
    no_hg_hostnames = defaultdict(list)
    all_orgs_hostnames = defaultdict(list)
    each_hg = defaultdict(list)
    for org, hostnames in best_hostnames_per_org.items():
        if org in HG_ORGS:
            hg_hostnames[org].extend(hostnames)

        if org not in HG_ORGS:
            no_hg_hostnames[org].extend(hostnames)

        all_orgs_hostnames[org].extend(hostnames)

        if org in HG_ORGS:
            each_hg[org].extend(hostnames)

    return hg_hostnames, no_hg_hostnames, all_orgs_hostnames, each_hg


def compute_score(ordered_hg: list = None) -> None:
    """calculate score for each organization/ns pair"""
    targets = load_targets(TARGES_TABLE)
    vps = load_vps(VPS_TABLE)

    hg_hostnames, no_hg_hostnames, all_orgs_hostnames, each_hg = load_hostnames()

    #############################################################################################################################
    logger.info("Calculating scores for each HG separately")
    for hg, hostnames in each_hg.items():
        logger.info(f"{hg}:: {len(each_hg)} hostnames")
        output_path = RESULTS_PATH / f"scores__{hg}.pickle"
        get_scores(
            output_path=output_path,
            hostnames=hostnames,
            target_subnets=[t["subnet"] for t in targets],
            vp_subnets=[v["subnet"] for v in vps],
            target_ecs_table=TARGETS_ECS_TABLE,
            vps_ecs_table=VPS_ECS_TABLE,
        )

    ##############################################################################################################################
    logger.info(f"All HG hostnames:: {len(hg_hostnames)} orgs")
    output_path = RESULTS_PATH / f"scores__hg_orgs.pickle"
    get_scores(
        output_path=output_path,
        hostnames=hg_hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )

    ##############################################################################################################################
    logger.info(f"No HG hostnames:: {len(no_hg_hostnames)} orgs")
    output_path = RESULTS_PATH / f"scores__no_hg_orgs.pickle"
    get_scores(
        output_path=output_path,
        hostnames=no_hg_hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )

    ##############################################################################################################################
    logger.info(f"All orgs hostnames:: {len(all_orgs_hostnames)} orgs")
    output_path = RESULTS_PATH / f"scores__all_orgs.pickle"
    get_scores(
        output_path=output_path,
        hostnames=all_orgs_hostnames,
        target_subnets=[t["subnet"] for t in targets],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=TARGETS_ECS_TABLE,
        vps_ecs_table=VPS_ECS_TABLE,
    )

    #############################################################################################################################
    if ordered_hg:
        # create a list of hostname selection (remove one new org each time)
        for i in range(len(ordered_hg) - 1):
            new_config_hostnames = {}
            # remove the n first hg
            for org, hostnames in hg_hostnames.items():
                if org in ordered_hg[: (i + 1)]:
                    continue
                new_config_hostnames[org] = hostnames

            # add config for score calculation
            logger.info(f"nb orgs for hostnames:: {len(new_config_hostnames.keys())}")

            # save config for checking
            removed_hg = "_".join([ordered_hg[j] for j in range(0, i + 1)])

            output_path = (
                path_settings.RESULTS_PATH
                / f"tier2_evaluation/hostnames__georesolver_minus_{removed_hg}.json"
            )

            hostnames = set()
            for org_hostnames in new_config_hostnames.values():
                hostnames.update(org_hostnames)

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

    targets = load_targets(ch_settings.VPS_FILTERED_TABLE)

    for score_file in RESULTS_PATH.iterdir():

        if "scores" not in score_file.name:
            continue

        if only_hgs:
            hg_name = score_file.name.split("scores__")[-1].split(".")[0]
            if not hg_name in HG_ORGS:
                continue

        output_file = (
            RESULTS_PATH / f"{'results' + str(score_file).split('scores')[-1]}"
        )

        scores = load_pickle(score_file)

        get_vp_selection_per_target(
            output_path=output_file,
            scores=scores,
            targets=[t["addr"] for t in targets],
        )


def plot() -> None:
    cdfs = []

    eval_files = {
        "results__all_orgs.pickle": "All (GeoResolver)",
        "results__hg_orgs.pickle": "Only Hypergiants",
        "results__no_hg_orgs.pickle": "No Hypergiants",
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
        output_path="hg_vs_no_hg_vs_all",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def order_hg() -> list:
    """order HG based on the fraction of IP addresses geolocated under 40km"""
    removed_vps = load_json(
        path_settings.DATASET / "imc2024_generated_files/removed_vps.json"
    )
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        removed_vps,
    )

    frac_under_40_per_hg = {}
    for i, file in enumerate(RESULTS_PATH.iterdir()):

        if "results" not in file.name:
            continue

        hg = file.name.split("_")[-1].split(".")[0]
        if hg not in HG_ORGS:
            continue

        logger.info(f"Evaluation for {hg}")

        vp_selection_per_target = load_pickle(RESULTS_PATH / file)

        d_errors = get_d_errors_georesolver(
            pings_per_target=pings_per_target,
            vp_selection_per_target=vp_selection_per_target,
            vps_coordinates=vps_coordinates,
        )

        x, y = ecdf(d_errors)
        frac_under_40km = round(get_proportion_under(x, y, 40), 2)
        logger.info(f"{hg} :: {frac_under_40km=}")

        frac_under_40_per_hg[hg] = frac_under_40km

    return frac_under_40_per_hg


if __name__ == "__main__":
    compute_scores = True
    evaluation = True
    make_figure = True

    if compute_scores:
        frac_under_40_per_hg = []
        do_order_hg: bool = False
        if do_order_hg:
            frac_under_40_per_hg = order_hg()
            # frac_under_40_per_hg = [
            #     ("CDN77", 0.69),
            #     ("AMAZON", 0.62),
            #     ("INCAPSULA", 0.6),
            #     ("FACEBOOK", 0.54),
            #     ("AKAMAI", 0.47),
            #     ("APPLE", 0.46),
            #     ("GOOGLE", 0.46),
            #     ("OVH", 0.35),
            #     ("ALIBABA-CN-NET", 0.23),
            # ]
            ordered_hg = [hg for hg, _ in frac_under_40_per_hg]

        compute_score(ordered_hg)

    if evaluation:
        evaluate()
    if make_figure:
        plot()
