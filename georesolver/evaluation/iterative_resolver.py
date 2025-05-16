"""evaluate georesolver performances while not using GPDNS resolver"""

import asyncio
import numpy as np

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import (
    get_tables,
    get_hostnames,
    load_vps,
    load_targets,
    get_pings_per_target_extended,
)
from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_vp_selection_per_target,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    get_proportion_under,
    plot_multiple_cdf,
)
from georesolver.common.files_utils import (
    load_json,
    dump_json,
    load_csv,
    dump_csv,
)
from georesolver.common.ip_addresses_utils import get_host_ip_addr, get_prefix_from_ip
from georesolver.common.utils import get_d_errors_ref, get_d_errors_georesolver
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

ITERATIVE_GEORESOLVER_PATH = (
    path_settings.HOSTNAME_FILES / "iterative_georesolver_hostnames.csv"
)
ITERATIVE_GEORESOLVER_PATH = (
    path_settings.HOSTNAME_FILES / "iterative_georesolver_hostnames_bgp_10_ns_3.csv"
)
ITERATIVE_HOSTNAMES_TABLE = "iterative_ecs_hostnames"
ITERATIVE_VPS_ECS_MAPPING_TABLE = ch_settings.VPS_ECS_MAPPING_TABLE + "_iterative"

TARGETS_TABLE: str = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE: str = ch_settings.VPS_FILTERED_FINAL_TABLE
RESULTS_PATH: Path = path_settings.RESULTS_PATH / "iterative_eval"


def get_iterative_ecs_hostnames(input_table: str) -> list[str]:
    """use ZDNS resolver to filter hostnames"""
    ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ecs_selected_hostnames.csv"

    tables = get_tables()

    logger.info(f"Iterative ECS hostnames table:: {input_table}")

    if input_table not in tables:
        # load hostnames per org per ns
        best_hostnames_per_org_per_ns = load_json(
            path_settings.HOSTNAME_FILES / "best_hostnames_per_org_per_ns.json"
        )

        ecs_hostnames = set()
        for ns, hostnames_per_org in best_hostnames_per_org_per_ns.items():
            for hostnames in hostnames_per_org.values():
                ecs_hostnames.update([h for _, h, _, _ in hostnames])

        logger.info(f"Found {len(ecs_hostnames)} hostnames supporting ECS")

        dump_csv(ecs_hostnames, ecs_hostnames_path)

        host_addr = get_host_ip_addr()
        host_subnet = get_prefix_from_ip(host_addr)

        asyncio.run(
            run_dns_mapping(
                subnets=[host_subnet],
                hostname_file=ecs_hostnames_path,
                output_table=input_table,
                itterative=True,
            )
        )

    # load hostnames from table
    iterative_ecs_hostnames = get_hostnames(input_table)

    logger.info(
        f"Retrieved {len(iterative_ecs_hostnames)} hostnames supporting ECS with ZDNS resolver"
    )

    return iterative_ecs_hostnames


def get_best_hostnames_per_ns_per_org_iterative(
    input_path: Path, iterative_ecs_hostnames: list[str]
) -> dict:
    """return iterative ECS hostnames with their geo score, hosting org and ns"""
    iterative_hostnames_per_org_per_ns = defaultdict(dict)
    if not input_path.exists():
        best_hostnames_per_org_per_ns = load_json(
            path_settings.HOSTNAME_FILES / "best_hostnames_per_org_per_ns.json"
        )
        for ns, hostnames_per_org in tqdm(best_hostnames_per_org_per_ns.items()):
            for org, hostnames in hostnames_per_org.items():
                for index, hostname, geo_score, nb_bgp_prefixes in hostnames:
                    # select hostnames present in original dataset
                    if hostname in iterative_ecs_hostnames:
                        try:
                            iterative_hostnames_per_org_per_ns[ns][org].append(
                                (index, hostname, geo_score, nb_bgp_prefixes)
                            )
                        except KeyError:
                            iterative_hostnames_per_org_per_ns[ns][org] = [
                                (index, hostname, geo_score, nb_bgp_prefixes)
                            ]

        dump_json(iterative_hostnames_per_org_per_ns, input_path)
        return iterative_hostnames_per_org_per_ns

    iterative_hostnames_per_org_per_ns = load_json(input_path)
    return iterative_hostnames_per_org_per_ns


def get_hostnames_org_and_ns_threshold(
    best_hostnames_per_org_per_ns: dict,
) -> list[str]:
    """
    filter hostnames with a sufficient number of returned BGP prefixes
    while preserving (NS/ORG) pair diversity
    """
    bgp_threshold = 10
    ns_org_threshold = 3

    filtered_hostnames_per_org_per_ns = defaultdict(dict)
    for ns in best_hostnames_per_org_per_ns:
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():

            hostnames = sorted(hostnames, key=lambda x: x[2])

            for index, hostname, _, nb_bgp_prefixes in hostnames:

                # filter hostnames with less than 10 BGP prefixes
                if nb_bgp_prefixes < bgp_threshold:
                    continue

                try:
                    if (
                        len(filtered_hostnames_per_org_per_ns[ns][org])
                        < ns_org_threshold
                    ):

                        filtered_hostnames_per_org_per_ns[ns][org].append(hostname)
                except KeyError:
                    filtered_hostnames_per_org_per_ns[ns][org] = [hostname]

    filtered_hostnames = set()
    filtered_hostnames_per_org = defaultdict(set)
    nb_pairs = set()
    nb_orgs = set()
    for ns in filtered_hostnames_per_org_per_ns:
        # logger.info(f"{ns}")
        for org, hostnames_score in filtered_hostnames_per_org_per_ns[ns].items():
            # logger.info(f"{org=}, {len(hostnames_score)}")
            filtered_hostnames.update([hostname for hostname in hostnames_score])
            filtered_hostnames_per_org[org].update(
                [hostname for hostname in hostnames_score]
            )
            nb_orgs.add(org)
            nb_pairs.add((ns, org))

    # type cast for json compatibility
    for org, hostnames in filtered_hostnames_per_org.items():
        filtered_hostnames_per_org[org] = list(hostnames)

    logger.info(
        f"Per NS/org:: {bgp_threshold=}, {ns_org_threshold=}:: {len(filtered_hostnames)=}"
    )
    logger.info(f"Nb Name servers:: {len(filtered_hostnames_per_org_per_ns)}")
    logger.info(f"Nb orgs:: {len(nb_orgs)}")
    logger.info(f"Nb pairs:: {len(nb_pairs)}")

    return (
        list(filtered_hostnames),
        filtered_hostnames_per_org,
        filtered_hostnames_per_org_per_ns,
    )


def get_iterative_georesolver_hostnames(input_path: Path) -> list[str]:
    """get a list of iterative ECS hostnames"""
    if not input_path.exists():
        iterative_hostnames_per_org_per_ns = (
            path_settings.HOSTNAME_FILES / "iterative_hostnames_per_org_per_ns.json"
        )

        # 1. ECS-DNS resolution: 1 subnet, best ECS hostnames
        iterative_ecs_hostnames = get_iterative_ecs_hostnames(ITERATIVE_HOSTNAMES_TABLE)

        # 2. Retrieve hostnames geo score, name server and hosting org
        iterative_hostnames_per_org_per_ns = (
            get_best_hostnames_per_ns_per_org_iterative(
                iterative_hostnames_per_org_per_ns, iterative_ecs_hostnames
            )
        )

        # 3. Filter hostnames based on:
        #   - returned number of BGP prefixes
        #   - number of hostnames per (NS/org) pairs
        (
            filtered_hostnames,
            filtered_hostnames_per_org,
            filtered_hostnames_per_org_per_ns,
        ) = get_hostnames_org_and_ns_threshold(iterative_hostnames_per_org_per_ns)

        # save data
        dump_csv(filtered_hostnames, input_path)
        dump_json(
            filtered_hostnames_per_org,
            path_settings.HOSTNAME_FILES / "iterative_filtered_hostnames_per_org.json",
        )
        dump_json(
            filtered_hostnames_per_org_per_ns,
            path_settings.HOSTNAME_FILES
            / "iterative_filtered_hostnames_per_org_per_ns.json",
        )


################################################################
# EVAL STEP
################################################################
def iterative_vps_ecs_mapping(hostname_file: Path, output_table: str) -> None:
    """perform vps ecs mapping"""
    tables = get_tables()
    if output_table not in tables:
        vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
        vps_subnets = [vp["subnet"] for vp in vps]
        asyncio.run(
            run_dns_mapping(
                subnets=vps_subnets,
                hostname_file=hostname_file,
                output_table=output_table,
                itterative=True,
            )
        )
    else:
        logger.info(f"Iterative VPs mapping {output_table=} already exists")


def plot_iterative_results() -> None:

    cdfs = []
    removed_vps = load_json(path_settings.REMOVED_VPS)
    hostnames = load_csv(
        path_settings.HOSTNAME_FILES / "iterative_georesolver_hostnames_bgp_10_ns_3.csv"
    )
    targets = load_targets(TARGETS_TABLE)
    vps = load_vps(VPS_TABLE)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    target_subnets = [t["subnet"] for t in targets]
    vp_subnets = [v["subnet"] for v in vps]

    # load score similarity between vps and targets
    scores = get_scores(
        output_path=RESULTS_PATH / "score_extended.pickle",
        hostnames=hostnames,
        target_subnets=target_subnets,
        vp_subnets=vp_subnets,
        target_ecs_table=ITERATIVE_VPS_ECS_MAPPING_TABLE,
        vps_ecs_table=ITERATIVE_VPS_ECS_MAPPING_TABLE,
    )

    vp_selection_per_target = get_vp_selection_per_target(
        output_path=RESULTS_PATH / "vp_selection_extended.pickle",
        scores=scores,
        targets=[t["addr"] for t in targets],
        vps=vps,
    )

    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE, removed_vps
    )

    # add reference
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    # add georesolver
    vp_selection_per_target_georesolver = get_vp_selection_per_target(
        output_path=path_settings.RESULTS_PATH / "tier5_evaluation/vp_selection.pickle",
        scores=scores,
        targets=[t["addr"] for t in targets],
        vps=vps,
    )
    d_errors = get_d_errors_georesolver(
        targets=[t["addr"] for t in targets],
        pings_per_target=pings_per_target,
        vp_selection_per_target=vp_selection_per_target_georesolver,
        vps_coordinates=vps_coordinates,
        probing_budget=50,
    )
    x, y = ecdf(d_errors)
    cdfs.append((x, y, "GPDNS resolver (GeoResolver)"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"GeoResolver:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"GeoResolver:: median_error={round(m_error, 2)} [km]")

    # add iterative resolver results
    d_errors = get_d_errors_georesolver(
        targets=[t["addr"] for t in targets],
        pings_per_target=pings_per_target,
        vp_selection_per_target=vp_selection_per_target,
        vps_coordinates=vps_coordinates,
        probing_budget=50,
    )
    x, y = ecdf(d_errors)
    cdfs.append((x, y, "ZDNS resolver"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"ZDNS resolver:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"ZDNS resolver:: median_error={round(m_error, 2)} [km]")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="iterative_resolver",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def main() -> None:
    """
    entry point:
        - perform ECS-DNS resolution on classified DNS hostnames (best hostnames)
        - make hostname selection based on new hostname set (using returned BGP prefixes)
        - perform VPs ECS mapping using new selected hostnames
        - evaluation on meshed pings
    """
    do_get_iterative_georesolver_hostnames: bool = False
    do_eval: bool = True

    if do_get_iterative_georesolver_hostnames:
        get_iterative_georesolver_hostnames(ITERATIVE_GEORESOLVER_PATH)

    if do_eval:
        # 0. perform VPs mapping
        iterative_vps_ecs_mapping(
            path_settings.HOSTNAMES_GEORESOLVER, ITERATIVE_VPS_ECS_MAPPING_TABLE
        )

        # 1. make cdfs
        plot_iterative_results()


if __name__ == "__main__":
    main()
