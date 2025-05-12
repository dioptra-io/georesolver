"""evaluate georesolver performances while not using GPDNS resolver"""

import asyncio

from tqdm import tqdm
from pyasn import pyasn
from pathlib import Path
from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import (
    get_tables,
    get_hostnames,
    load_vps,
    load_targets,
    get_min_rtt_per_vp,
    get_pings_per_target,
)
from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.evaluation.evaluation_score_functions import get_scores
from georesolver.evaluation.evaluation_plot_functions import (
    plot_ref,
    ecdf,
    plot_multiple_cdf,
)
from georesolver.evaluation.evaluation_ecs_geoloc_functions import (
    ecs_dns_vp_selection_eval,
)
from georesolver.common.utils import EvalResults, TargetScores, get_parsed_vps
from georesolver.common.files_utils import (
    load_json,
    dump_json,
    load_csv,
    dump_csv,
    load_pickle,
    dump_pickle,
)
from georesolver.common.ip_addresses_utils import get_host_ip_addr, get_prefix_from_ip
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

ITERATIVE_GEORESOLVER_PATH = (
    path_settings.HOSTNAME_FILES / "iterative_georesolver_hostnames.csv"
)
ITERATIVE_GEORESOLVER_PATH = (
    path_settings.HOSTNAME_FILES / "iterative_georesolver_hostnames_bgp_10_ns_5.csv"
)
ITERATIVE_HOSTNAMES_TABLE = "iterative_ecs_hostnames"
ITERATIVE_VPS_ECS_MAPPING_TABLES = ch_settings.VPS_ECS_MAPPING_TABLE + "_iterative"
ITERATIVE_VPS_ECS_MAPPING_TABLES = ch_settings.VPS_ECS_MAPPING_TABLE + "_iterative_new"


def get_cisco_ecs_hostnames(input_table: str) -> list[str]:
    """get ECS hostnames using Cisco open DNS"""
    host_addr = get_host_ip_addr()
    host_subnet = get_prefix_from_ip(host_addr)
    ecs_hostnames_path = path_settings.HOSTNAMES_GEORESOLVER

    asyncio.run(
        run_dns_mapping(
            subnets=[host_subnet],
            hostname_file=ecs_hostnames_path,
            output_table=input_table,
            # itterative=True,
        )
    )


def get_iterative_ecs_hostnames(input_table: str) -> list[str]:
    """use ZDNS resolver to filter hostnames"""
    ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ecs_hostnames_GPDNS.csv"

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
    ns_org_threshold = 5

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


def compute_score(
    output_path: Path, hostname_file: Path, vps_ecs_mapping_table: str
) -> None:
    """calculate score for each organization/ns pair"""

    # if output_path.exists():
    #     return

    targets_table = ch_settings.VPS_FILTERED_FINAL_TABLE
    vps_table = ch_settings.VPS_FILTERED_FINAL_TABLE
    targets_ecs_table = vps_ecs_mapping_table
    vps_ecs_table = vps_ecs_mapping_table

    selected_hostnames = load_csv(hostname_file)

    score_config = {
        "targets_table": targets_table,
        "main_org_threshold": 0.0,
        "bgp_prefixes_threshold": 0.0,
        "vps_table": vps_table,
        "selected_hostnames": selected_hostnames,
        "targets_ecs_table": targets_ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": ["jaccard"],
        "answer_granularities": ["answer_subnets"],
        "output_path": output_path,
    }

    get_scores(score_config)


def select_vps(score_file: Path, output_file: Path) -> None:
    """perform georesolver vp selection based on score calculation"""

    # if output_file.exists():
    #     return

    asndb = pyasn(str(path_settings.RIB_TABLE))

    logger.info(f"Running geresolver analysis from score file:: {score_file}")

    removed_vps = load_json(path_settings.REMOVED_VPS)
    targets = load_targets(ch_settings.VPS_FILTERED_FINAL_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)
    last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    ping_vps_to_target = get_pings_per_target(
        ch_settings.VPS_MESHED_PINGS_TABLE, [addr for _, addr in removed_vps]
    )

    vps_per_addr = {}
    for vp in vps:
        vps_per_addr[vp["addr"]] = vp

    scores: TargetScores = load_pickle(score_file)

    results_answer_subnets = ecs_dns_vp_selection_eval(
        targets=targets,
        vps_per_subnet=vps_per_subnet,
        subnet_scores=scores.score_answer_subnets,
        ping_vps_to_target=ping_vps_to_target,
        last_mile_delay=last_mile_delay,
        vps_coordinates=vps_coordinates,
        vps_per_addr=vps_per_addr,
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


def plot_iterative_results(
    georesolver_results_path: Path,
    iterative_results_path: Path,
    output_path: Path,
    metric_evaluated: str = "d_error",
    legend_pos: str = "lower right",
) -> None:
    all_cdfs = []

    # exhaustive curve
    ref_cdf = plot_ref(metric_evaluated)
    all_cdfs.append(ref_cdf)

    # georesolver results
    d_errors = []
    georesolver_results: EvalResults = load_pickle(georesolver_results_path)
    for _, target_results in georesolver_results.results_answer_subnets.items():
        try:
            results = target_results["result_per_metric"]["jaccard"]
            d_errors.append(results["ecs_shortest_ping_vp_per_budget"][50]["d_error"])

        except KeyError:
            continue

    x, y = ecdf(d_errors)
    all_cdfs.append((x, y, "GPDNS resolver (GeoResolver)"))

    # iterative results
    d_errors = []
    iterative_results: EvalResults = load_pickle(iterative_results_path)
    for _, target_results in iterative_results.results_answer_subnets.items():
        try:
            results = target_results["result_per_metric"]["jaccard"]
            d_errors.append(results["ecs_shortest_ping_vp_per_budget"][50]["d_error"])
        except KeyError:
            continue

    x, y = ecdf(d_errors)
    all_cdfs.append((x, y, "ZDNS resolver"))

    plot_multiple_cdf(
        cdfs=all_cdfs,
        output_path=output_path,
        metric_evaluated=metric_evaluated,
        legend_pos=legend_pos,
    )


def evaluation(hostname_file: Path, vps_ecs_mapping_table: str) -> None:
    """calculate scores, select VPs and eval against meshed pings"""
    results_path = path_settings.RESULTS_PATH / "iterative_eval"
    georesolver_results_path = (
        path_settings.RESULTS_PATH
        / "tier5_evaluation/results__d_error_per_budget_new.pickle"
    )

    # 0. perform VPs mapping
    iterative_vps_ecs_mapping(hostname_file, vps_ecs_mapping_table)

    # 1. compute scores
    compute_score(
        output_path=results_path / "score_new.pickle",
        hostname_file=hostname_file,
        vps_ecs_mapping_table=vps_ecs_mapping_table,
    )

    # 2. select VPs
    select_vps(
        score_file=results_path / "score_new.pickle",
        output_file=results_path / "vp_selection_new.pickle",
    )

    # 3. make cdfs
    plot_iterative_results(
        georesolver_results_path=georesolver_results_path,
        iterative_results_path=results_path / "vp_selection_new.pickle",
        output_path="iterative_resolver_evaluation_new",
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
        evaluation(
            path_settings.HOSTNAMES_GEORESOLVER, ITERATIVE_VPS_ECS_MAPPING_TABLES
        )


if __name__ == "__main__":
    main()
