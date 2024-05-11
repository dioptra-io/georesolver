import asyncio
import json

from datetime import datetime
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from geogiant.prober.ripe_api import RIPEAtlasAPI
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.evaluation.plot import plot_ripe_ip_map
from geogiant.evaluation.ecs_geoloc_eval import get_shortest_ping_geo_resolver
from geogiant.common.geoloc import rtt_to_km
from geogiant.common.queries import get_pings_per_target, load_vps, get_measurement_ids
from geogiant.common.utils import get_shortest_ping_all_vp, get_random_shortest_ping
from geogiant.common.files_utils import (
    load_json,
    load_pickle,
    dump_pickle,
    dump_json,
    load_csv,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings, ConstantSettings

path_settings = PathSettings()
constant_settings = ConstantSettings()
clickhouse_settings = ClickhouseSettings()

# INPUT TABLE
PING_RIPE_ATLAS_TABLE = "pings_ripe_altas"
PING_RIPE_IP_MAP_TABLE = "pings_ripe_ip_map"

# OUTPUT TABLES
ECS_TABLE = "end_to_end_mapping_ecs"
VPS_ECS_TABLE = "vps_mapping_ecs"
PING_END_TO_END_TABLE = "pings_end_to_end"

# FILE PATHS
END_TO_END_HOSTNAMES_PATH = (
    path_settings.END_TO_END_DATASET / "end_to_end_hostnames.csv"
)
RIPE_IP_MAP_TARGETS_PATH = (
    path_settings.END_TO_END_DATASET / "ripe_ip_map_targets_evaluation.json"
)
RIPE_IP_MAP_SUBNETS_PATH = path_settings.END_TO_END_DATASET / "ripe_ip_map_subnets.json"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"
RIPE_IP_MAP_EVALUATION_PATH = (
    path_settings.RESULTS_PATH / "ripe_ip_map_evaluation/results__ripe_ip_map.pickle"
)
RIPE_IP_MAP_GEO_INFO = path_settings.END_TO_END_DATASET / "ripe_ip_map_geo_info.json"


# CONSTANT PARAMETERS
latency_threshold = 50_000
probing_budget = 500
max_nb_measurements = 1_000

# START_DATE: str = "2024-04-19"
# END_DATE: str = "2024-04-18"
# START_DATE: str = "2024-04-16"
# END_DATE: str = "2024-04-17"
START_DATE: str = "2024-04-15"
END_DATE: str = "2024-04-16"


async def insert_results_from_tag() -> None:
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_time = datetime.strptime(END_DATE, "%Y-%m-%d")
    start_time = int(start_time.timestamp())
    end_time = int(end_time.timestamp())

    params = {
        "tags": "single-radius",
        "type": "ping",
        "af": 4,
        "start_time__gte": start_time,
        "stop_time__gte": end_time,
    }

    logger.info(f"Retreiving measurement results:: {start_time} - {end_time}")
    await RIPEAtlasAPI().get_results_from_tag(
        params=params, ping_table=PING_RIPE_IP_MAP_TABLE
    )


def get_ripe_ip_map_score() -> None:
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
                / f"ripe_ip_map_evaluation/scores__best_hostname_geo_score_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.pickle"
            )

            # some organizations do not have enought hostnames
            if output_path.exists():
                logger.info(
                    f"Score for {bgp_threshold} BGP prefix threshold alredy done"
                )
                continue

            score_config = {
                "targets_subnet_path": RIPE_IP_MAP_SUBNETS_PATH,
                "vps_subnet_path": VPS_SUBNET_PATH,
                "hostname_per_cdn": selected_hostnames_per_cdn,
                "selected_hostnames": selected_hostnames,
                "targets_ecs_table": ECS_TABLE,
                "vps_ecs_table": VPS_ECS_TABLE,
                "hostname_selection": "max_bgp_prefix",
                "score_metric": ["jaccard"],
                "answer_granularities": ["answer_subnets"],
                "output_path": output_path,
            }

            get_scores(score_config)


def get_vps_pings(
    target_associated_vps: list,
    ping_to_target: list,
) -> list:
    vp_selection = []

    # filter out all vps not included by ecs-dns methodology
    for vp_addr, min_rtt in ping_to_target:
        if vp_addr in target_associated_vps:
            vp_selection.append((vp_addr, min_rtt))

    return vp_selection


def get_ripe_ip_map_shortest_ping(targets: list[str]) -> None:
    """for each IP addresses retrieved from RIPE IP map, get the shortest ping"""
    pings_single_radius = get_pings_per_target(PING_RIPE_IP_MAP_TABLE)

    shortest_ping_per_target = []
    for target, pings in pings_single_radius.items():
        if target in targets:
            _, shortest_ping_rtt = min(pings, key=lambda x: x[-1])
            shortest_ping_per_target.append((target, shortest_ping_rtt))

    return shortest_ping_per_target


def detect_anycast(
    ping_vps_to_target: list,
    vp_distance_matrix: dict[dict],
) -> bool:
    """detect if an IP address is anycast or not based on measured pings"""
    soi = []
    for vp_i, min_rtt_i in ping_vps_to_target:
        for vp_j, min_rtt_j in ping_vps_to_target:
            if vp_i == vp_i:
                continue

            vps_distance = vp_distance_matrix[vp_i][vp_j]
            cumulative_rtt_dist = rtt_to_km(min_rtt_i) + rtt_to_km(min_rtt_j)

            if vps_distance > cumulative_rtt_dist:
                soi.append((vp_i, vp_j))

    if soi:
        return True
    else:
        return False


def filter_on_latency(
    geo_resolver_sp, ref_sp, ripe_ip_map_sp, latency_threshold
) -> None:
    filtered_targets = set()
    filtered_ref_sp = []
    for target, min_rtt in ref_sp:
        if min_rtt < latency_threshold:
            filtered_targets.add(target)
            filtered_ref_sp.append((target, min_rtt))

    filtered_geo_resolver_sp = defaultdict(list)
    for probing_budget, results in geo_resolver_sp.items():
        for target, _, min_rtt in results:
            if target in filtered_targets:
                filtered_geo_resolver_sp[probing_budget].append((target, min_rtt))

    filtered_ripe_ip_map_sp = []
    for target, min_rtt in ripe_ip_map_sp:
        if target in filtered_targets:
            filtered_ripe_ip_map_sp.append((target, min_rtt))

    logger.info(
        f"{len(filtered_geo_resolver_sp[50])} IP addresses remaining under {latency_threshold}ms"
    )

    return filtered_geo_resolver_sp, filtered_ripe_ip_map_sp, ref_sp


def evaluate() -> None:
    """calculate distance error and latency for each score"""

    targets = load_json(RIPE_IP_MAP_TARGETS_PATH)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    removed_vps = load_json(path_settings.REMOVED_VPS)

    score_files = [
        path_settings.RESULTS_PATH
        / "ripe_ip_map_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    ]

    logger.info("RIPE IP MAP:: GeoResolver evaluation")
    geo_resolver_sp = {}
    for score_file in score_files:
        bgp_prefix_threshold = int(score_file.name.split("score_")[-1].split("_")[0])
        nb_hostnames_per_org_ns = int(score_file.name.split("BGP_")[-1].split("_")[0])

        shortest_ping_per_target = get_shortest_ping_geo_resolver(
            targets=targets,
            vps=vps,
            score_file=score_file,
            ping_table=PING_END_TO_END_TABLE,
            removed_vps=removed_vps,
            probing_budgets=[50, 100, 500],
        )

        geo_resolver_sp = shortest_ping_per_target

        targets = [target for target, _, _ in geo_resolver_sp[50]]

    logger.info("RIPE IP MAP:: Single-radius evaluation")
    ripe_ip_map_sp = get_shortest_ping_all_vp(
        targets, PING_RIPE_IP_MAP_TABLE, removed_vps
    )

    logger.info("Random 50 VPs")
    random_sp = get_random_shortest_ping(targets, PING_END_TO_END_TABLE, removed_vps)

    logger.info("RIPE IP MAP:: Reference evaluation")
    ref_sp = get_shortest_ping_all_vp(targets, PING_END_TO_END_TABLE, removed_vps)

    # filter_on_latency(geo_resolver_sp, ref_sp, ripe_ip_map_sp, 2)

    dump_pickle(
        (geo_resolver_sp, ripe_ip_map_sp, ref_sp, random_sp),
        RIPE_IP_MAP_EVALUATION_PATH,
    )


def calculate_percentage_of_occurrence(lst: list):
    total_elements = len(lst)
    percentage_occurrence = {}
    keys = set(lst)
    for key in keys:
        nb_occurence = lst.count(key)
        percentage_occurrence[key] = round((nb_occurence / total_elements) * 100, 2)

    return percentage_occurrence


def get_measurement_overhead() -> None:
    measurement_ids = get_measurement_ids(PING_RIPE_IP_MAP_TABLE)

    logger.debug(f"Measurement overhead on:: {len(measurement_ids)}")

    measurement_overhead = {}
    count = 0
    for measurement_id in tqdm(measurement_ids):
        probe_requested, dst_addr = RIPEAtlasAPI().get_probe_requested(measurement_id)
        measurement_overhead[dst_addr] = probe_requested

        count += 1
        if count > 100:
            dump_json(
                measurement_overhead,
                path_settings.END_TO_END_DATASET
                / "measurement_overhead_ripe_ip_map.json",
            )
            count = 0

    dump_json(
        measurement_overhead,
        path_settings.END_TO_END_DATASET / "measurement_overhead_ripe_ip_map.json",
    )


async def main() -> None:
    retrieve_public_measurements = False
    calculate_score = False
    evaluation = False
    make_figures = False
    overhead = True

    if retrieve_public_measurements:
        await insert_results_from_tag()

    if calculate_score:
        get_ripe_ip_map_score()

    if evaluation:
        evaluate()

    if make_figures:
        geo_resolver_sp, ripe_ip_map_sp, ref_sp, random_sp = load_pickle(
            RIPE_IP_MAP_EVALUATION_PATH
        )
        plot_ripe_ip_map(
            geo_resolver_sp,
            ripe_ip_map_sp,
            ref_sp,
            random_sp,
            output_path="ripe_ip_map_evaluation",
        )

        # check country ip addresses
        targets = load_csv(
            path_settings.END_TO_END_DATASET / "current_ripe_ip_map_targets.csv"
        )
        targets_country = {}
        with RIPE_IP_MAP_GEO_INFO.open() as f:
            for i, row in enumerate(f.readlines()):
                row = json.loads(row)
                target_addr = targets[i]
                country = row["country"]
                targets_country[target_addr] = country

        countries = []
        geo_resolver_cn_target_rtts = []
        for target, _, min_rtt in geo_resolver_sp[50]:
            try:
                country = targets_country[target]
            except KeyError:
                continue

            if country == "CN":
                geo_resolver_cn_target_rtts.append(min_rtt)

            countries.append(country)

        ref_cn_target_rtts = []
        for target, min_rtt in ref_sp:
            try:
                country = targets_country[target]
            except KeyError:
                continue

            if country == "CN":
                ref_cn_target_rtts.append(min_rtt)

        percentage_per_country = calculate_percentage_of_occurrence(countries)
        percentage_per_country = sorted(
            percentage_per_country.items(), key=lambda x: x[1]
        )

        logger.info("Percentage of occurrence for each value:")
        for value, percentage in percentage_per_country:
            logger.info(f"{value}: {percentage}%")

        cn_proportiob_under = (
            len([rtt for rtt in geo_resolver_cn_target_rtts if min_rtt <= 2])
            / len(geo_resolver_sp)
            * 100
        )
        logger.info(
            f"Geo resolver CN targets under 2ms:: {round(cn_proportiob_under, 2)}%"
        )

        cn_proportiob_under = (
            len([rtt for rtt in ref_cn_target_rtts if min_rtt <= 2]) / len(ref_sp) * 100
        )
        logger.info(f"Ref CN targets under 2ms:: {round(cn_proportiob_under, 2)}%")

    if overhead:
        overhead = get_measurement_overhead()


if __name__ == "__main__":
    asyncio.run(main())
