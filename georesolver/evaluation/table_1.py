import numpy as np

from loguru import logger
from collections import defaultdict, OrderedDict

from georesolver.clickhouse.queries import (
    get_pings_per_target_extended,
    load_targets,
    load_vps,
)
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_d_errors_georesolver,
    get_vp_selection_per_target,
)
from georesolver.evaluation.evaluation_plot_functions import ecdf, get_proportion_under
from georesolver.common.files_utils import load_json, load_pickle
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

TARGETS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
PING_TABLE = "vps_meshed_pings_CoNEXT_summer_submision"
VPS_ECS_TABLE = "vps_ecs_mapping__2025_04_13"
RESULTS_PATH = path_settings.RESULTS_PATH / "table_1"


def get_bgp_prefixes_per_hostname(cdn_per_hostname: dict) -> dict:
    """return the number of unique bgp prefixes per hostname"""
    bgp_prefix_per_hostname = defaultdict(set)
    for hostname, bgp_prefixes_per_cdn in cdn_per_hostname.items():
        for bgp_prefixes in bgp_prefixes_per_cdn.values():
            bgp_prefix_per_hostname[hostname].update(bgp_prefixes)

    return bgp_prefix_per_hostname


def compute_score() -> None:
    """calculate score for each organization/ns pair"""
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)

    for bgp_threshold in [5, 10, 20, 50, 100]:
        for org_ns_threshold in [3, 5, 10]:

            selected_hostnames_per_cdn_per_ns = load_json(
                path_settings.HOSTNAME_FILES
                / f"hostname__{bgp_threshold}_BGP_{org_ns_threshold}_org_ns.json"
            )

            selected_hostnames = set()
            selected_hostnames_per_cdn = defaultdict(list)
            for ns in selected_hostnames_per_cdn_per_ns:
                for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
                    selected_hostnames.update(hostnames)
                    selected_hostnames_per_cdn[org].extend(hostnames)

            logger.info(f"{len(selected_hostnames)=}")

            output_path = (
                RESULTS_PATH
                / f"scores__{bgp_threshold}_BGP_{org_ns_threshold}_org_ns.pickle"
            )

            get_scores(
                output_path=output_path,
                hostnames=selected_hostnames,
                target_subnets=[t["subnet"] for t in targets],
                vp_subnets=[v["subnet"] for v in vps],
                target_ecs_table=VPS_ECS_TABLE,
                vps_ecs_table=VPS_ECS_TABLE,
            )


def evaluation() -> None:
    """calculate distance error and latency for each score"""
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)

    logger.info("BGP prefix score geoloc evaluation")

    for score_file in RESULTS_PATH.iterdir():

        if "results" in score_file.name:
            continue

        logger.info(f"ECS evaluation for score:: {score_file}")
        scores = load_pickle(score_file)
        output_path = (
            RESULTS_PATH / f"{'results' + str(score_file).split('scores')[-1]}"
        )

        get_vp_selection_per_target(
            output_path=output_path,
            scores=scores,
            targets=[t["addr"] for t in targets],
            vps=vps,
        )


def plot() -> None:
    removed_vps = load_json(path_settings.REMOVED_VPS)
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    pings_per_target = get_pings_per_target_extended(PING_TABLE, removed_vps)

    vp_selection_files = defaultdict(list)
    for file in RESULTS_PATH.iterdir():
        if "results" in file.name and "org_ns" in file.name:
            bgp_prefix_threshold = file.name.split("results__")[-1].split("_")[0]
            org_ns_threshold = file.name.split("BGP_")[-1].split("_")[0]
            vp_selection_files[int(bgp_prefix_threshold)].append(
                (int(org_ns_threshold), file)
            )

    for bgp_prefix_threshold in vp_selection_files:
        vp_selection_files[bgp_prefix_threshold] = sorted(
            vp_selection_files[bgp_prefix_threshold], key=lambda x: x[0]
        )

    vp_selection_files = OrderedDict(
        sorted(vp_selection_files.items(), key=lambda x: x[0])
    )

    for bgp_threshold in vp_selection_files:
        for org_ns_threshold, file in vp_selection_files[bgp_prefix_threshold]:

            logger.info(f"{bgp_prefix_threshold=}; {org_ns_threshold=}")

            hostnames_per_org_per_ns = load_json(
                path_settings.HOSTNAME_FILES
                / f"hostname__{bgp_threshold}_BGP_{org_ns_threshold}_org_ns.json"
            )

            vp_selection_per_target = load_pickle(file)

            d_errors = get_d_errors_georesolver(
                targets=[t["addr"] for t in targets],
                pings_per_target=pings_per_target,
                vp_selection_per_target=vp_selection_per_target,
                vps_coordinates=vps_coordinates,
            )

            x, y = ecdf(d_errors)
            m_error = round(np.median(x), 2)
            proportion_of_ip = get_proportion_under(x, y)

            hostnames = set()
            org_ns_pairs = set()
            for ns, hostnames_per_org in hostnames_per_org_per_ns.items():
                for org, h in hostnames_per_org.items():
                    hostnames.update(h)
                    org_ns_pairs.add((ns, org))

            logger.info("##################################################")
            logger.info(f"{bgp_threshold=}; {org_ns_threshold=}")
            logger.info(f"{len(hostnames)=}")
            logger.info(f"{len(org_ns_pairs)=}")
            logger.info(f"frac <40km={round(proportion_of_ip, 2)}")
            logger.info(f"median_error={round(m_error, 2)} [km]")
            logger.info("##################################################")


def main(
    do_compute_score=True,
    do_evaluation=True,
    do_plot=True,
) -> None:
    """
    reproduce Million Scale paper main results
    Once 10 best VPs found for a single IP, can extend to all /24
    further reducing measurement cost
    """
    if do_compute_score:
        compute_score()

    if do_evaluation:
        evaluation()

    if do_plot:
        plot()


if __name__ == "__main__":
    main()
