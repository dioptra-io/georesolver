import asyncio

from pyasn import pyasn
from collections import defaultdict
from loguru import logger

from geogiant.evaluation.plot import plot_routers
from geogiant.evaluation.scores import get_scores
from geogiant.evaluation.ecs_geoloc_utils import get_shortest_ping_geo_resolver
from geogiant.clickhouse.queries import load_vps
from geogiant.common.ip_addresses_utils import route_view_bgp_prefix
from geogiant.common.utils import get_shortest_ping_all_vp, get_random_shortest_ping
from geogiant.common.files_utils import load_json, dump_json, load_pickle, dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# CLICKHOUSE TABLE
PING_TABLE = "pings_end_to_end"
ECS_TABLE = "end_to_end_ecs_mapping"
VPS_ECS_MAPPING_TABLE = "vps_ecs_mapping"

# FILE PATHS
ROUTERS_TARGET_PATH = (
    path_settings.END_TO_END_DATASET / "routers_targets_evaluation.json"
)
ROUTERS_SUBNET_PATH = (
    path_settings.END_TO_END_DATASET / "routers_subnets_evaluation.json"
)
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"
ROUTERS_EVALUATION_PATH = (
    path_settings.RESULTS_PATH / "routers_evaluation/results__routers.pickle"
)

ROUTERS_EVALUATION_NO_USERS_PATH = (
    path_settings.RESULTS_PATH / "routers_evaluation/results__routers_no_users.pickle"
)

# CONSTANT PARAMETERS
probing_budget = 50


def get_routers_score() -> None:
    for bgp_threshold in [20, 50]:
        for nb_hostname_per_ns_org in [3]:

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

            logger.info(
                f"{bgp_threshold=}, {nb_hostname_per_ns_org}, {len(selected_hostnames)=}"
            )

            output_path = (
                path_settings.RESULTS_PATH
                / f"routers_evaluation/scores__best_hostname_geo_score_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.pickle"
            )

            # some organizations do not have enought hostnames
            if output_path.exists():
                logger.info(
                    f"Score for {bgp_threshold} BGP prefix threshold alredy done"
                )
                continue

            score_config = {
                "targets_subnet_path": ROUTERS_SUBNET_PATH,
                "vps_subnet_path": VPS_SUBNET_PATH,
                "hostname_per_cdn": selected_hostnames_per_cdn,
                "selected_hostnames": selected_hostnames,
                "targets_ecs_table": ECS_TABLE,
                "vps_ecs_table": VPS_ECS_MAPPING_TABLE,
                "hostname_selection": "max_bgp_prefix",
                "score_metric": ["jaccard"],
                "answer_granularities": ["answer_subnets"],
                "output_path": output_path,
            }

            get_scores(score_config)


def evaluate() -> None:
    """calculate distance error and latency for each score"""

    targets = load_json(ROUTERS_TARGET_PATH)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)

    score_files = [
        path_settings.RESULTS_PATH
        / "routers_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    ]

    geo_resolver_sp = {}
    for score_file in score_files:
        bgp_prefix_threshold = score_file.name.split("score_")[-1].split("_")[0]
        nb_hostnames_per_org_ns = score_file.name.split("BGP_")[-1].split("_")[0]

        geo_resolver_sp = get_shortest_ping_geo_resolver(
            targets, vps, score_file, PING_TABLE, removed_vps
        )

        targets = [target for target, _, _ in geo_resolver_sp[50]]

    random_sp = get_random_shortest_ping(targets, PING_TABLE, removed_vps)
    ref_sp = get_shortest_ping_all_vp(targets, PING_TABLE, removed_vps)

    dump_pickle((geo_resolver_sp, ref_sp, random_sp), ROUTERS_EVALUATION_PATH)

    return geo_resolver_sp, ref_sp


def evaluate_no_users() -> None:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    targets = load_json(ROUTERS_TARGET_PATH)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)

    if not (path_settings.DATASET / "parsed_aspop.json").exists():
        aspop = load_json(path_settings.DATASET / "aspop.json")
        parsed_aspop = {}
        for row in aspop["Data"]:
            asn = row["AS"]
            pop = row["Users"]

            if pop > 0:
                parsed_aspop[int(asn)] = pop

        dump_json(parsed_aspop, path_settings.DATASET / "parsed_aspop.json")

    parsed_aspop = load_json(path_settings.DATASET / "parsed_aspop.json")
    as_with_users = [int(asn) for asn in parsed_aspop]
    filtered_targets = []
    logger.debug(f"{len(targets)=}")
    for target in targets:
        asn, _ = route_view_bgp_prefix(target, asndb)
        if asn in as_with_users:
            continue

        filtered_targets.append(target)

    targets = filtered_targets
    logger.debug(f"{len(filtered_targets)=}")

    score_files = [
        path_settings.RESULTS_PATH
        / "routers_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    ]

    geo_resolver_sp = {}
    for score_file in score_files:
        bgp_prefix_threshold = score_file.name.split("score_")[-1].split("_")[0]
        nb_hostnames_per_org_ns = score_file.name.split("BGP_")[-1].split("_")[0]

        geo_resolver_sp = get_shortest_ping_geo_resolver(
            targets, vps, score_file, PING_TABLE, removed_vps
        )

        targets = [target for target, _, _ in geo_resolver_sp[50]]

    random_sp = get_random_shortest_ping(targets, PING_TABLE, removed_vps)
    ref_sp = get_shortest_ping_all_vp(targets, PING_TABLE, removed_vps)

    dump_pickle((geo_resolver_sp, ref_sp, random_sp), ROUTERS_EVALUATION_NO_USERS_PATH)

    return geo_resolver_sp, ref_sp


async def main() -> None:
    calculate_score = False
    evaluation = True
    make_figures = True

    if calculate_score:
        get_routers_score()

    if evaluation:
        evaluate_no_users()
        evaluate()

    if make_figures:
        geo_resolver_sp, ref_sp, random_sp = load_pickle(ROUTERS_EVALUATION_PATH)
        plot_routers(
            geo_resolver_sp, ref_sp, random_sp, output_path="routers_evaluation"
        )

        geo_resolver_under_2ms = []
        for target, _, min_rtt in geo_resolver_sp[50]:
            if min_rtt < 2:
                geo_resolver_under_2ms.append(target)

        ref_under_2ms = []
        for target, min_rtt in ref_sp:
            if min_rtt < 2:
                ref_under_2ms.append(target)

        proportion = len(geo_resolver_under_2ms) / len(ref_under_2ms) * 100
        logger.info(f"Proportion:: {round(proportion, 2)}%")

        logger.warning("NO USERS AS EVALUATION::")
        geo_resolver_sp, ref_sp, random_sp = load_pickle(
            ROUTERS_EVALUATION_NO_USERS_PATH
        )
        plot_routers(
            geo_resolver_sp,
            ref_sp,
            random_sp,
            output_path="routers_evaluation_no_users",
        )

        geo_resolver_under_2ms = []
        for target, _, min_rtt in geo_resolver_sp[50]:
            if min_rtt < 2:
                geo_resolver_under_2ms.append(target)

        ref_under_2ms = []
        for target, min_rtt in ref_sp:
            if min_rtt < 2:
                ref_under_2ms.append(target)

        proportion = len(geo_resolver_under_2ms) / len(ref_under_2ms) * 100
        logger.info(f"Proportion:: {round(proportion, 2)}%")


if __name__ == "__main__":
    asyncio.run(main())
