import asyncio

from pathlib import Path
from random import shuffle
from datetime import datetime
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pych_client import ClickHouseClient

from geogiant.clickhouse import GetDstPrefix, GetSubnets
from geogiant.prober.ripe_api import RIPEAtlasAPI
from geogiant.prober.ripe_prober import RIPEAtlasProber
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.evaluation.plot import plot_ripe_ip_map
from geogiant.evaluation.ecs_geoloc_eval import get_ecs_vps
from geogiant.common.geoloc import rtt_to_km
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    get_pings_per_target,
    load_vps,
    get_subnets,
    retrieve_pings,
)
from geogiant.common.utils import EvalResults, TargetScores, get_parsed_vps
from geogiant.common.files_utils import (
    load_csv,
    load_json,
    dump_json,
    load_pickle,
    dump_pickle,
    dump_csv,
    load_anycatch_data,
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
PING_RIPE_IP_MAP_GEO_RESOLVER = "ping_ripe_ip_map_geo_resolver"

# FILE PATHS
RIPE_IP_MAP_SUBNETS = path_settings.END_TO_END_DATASET / "ripe_ip_map_subnets.json"
ROUTERS_SUBNETS = path_settings.END_TO_END_DATASET / "routers_subnets.json"
END_TO_END_HOSTNAMES_PATH = (
    path_settings.END_TO_END_DATASET / "end_to_end_hostnames.csv"
)
END_TO_END_TARGETS_PATH = path_settings.END_TO_END_DATASET / "end_to_end_targets.json"
END_TO_END_SUBNETS_PATH = path_settings.END_TO_END_DATASET / "end_to_end_subnets.json"
RIPE_IP_MAP_SUBNETS_PATH = path_settings.END_TO_END_DATASET / "ripe_ip_map_subnets.json"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnet.json"


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


def get_ripe_ip_map_dst_prefix(latency_threshold: int) -> list:
    """retrieve all RIPE IP map subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetDstPrefix().execute(
            client=client,
            table_name=PING_RIPE_IP_MAP_TABLE,
            latency_threshold=latency_threshold,
        )

    ripe_ip_map_subnets = []
    for row in rows:
        ripe_ip_map_subnets.append(row["dst_prefix"])

    return ripe_ip_map_subnets


def get_subnets_from_pings(use_cache: bool = False) -> list[str]:
    ripe_ip_map_subnets = get_ripe_ip_map_dst_prefix(latency_threshold)

    if use_cache:
        resolved_subnets = get_subnets(ecs_table)
        if resolved_subnets:
            remaining_subnets = set(resolved_subnets).symmetric_difference(
                set(ripe_ip_map_subnets)
            )
            ripe_ip_map_subnets = remaining_subnets

    logger.info(
        f"Retrieved:: {len(ripe_ip_map_subnets)} RIPE IP map subnets, {latency_threshold=}"
    )
    dump_json(ripe_ip_map_subnets, ripe_ip_map_subnets_path)


def get_subnets_from_ecs() -> list[str]:
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetSubnets().execute(
            client=client,
            table_name=ecs_table,
        )

    subnets = []
    for row in rows:
        subnets.append(row["subnet"])

    return subnets


async def resolve_subnets() -> None:
    selected_hostnames = load_csv(
        path_settings.END_TO_END_DATASET / "end_to_end_hostnames.csv"
    )

    logger.info(f"Nb total hostnames:: {len(selected_hostnames)}")

    get_subnets_from_pings()

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=ripe_ip_map_subnets_path,
        output_table=ecs_table,
        chunk_size=100,
    )


def get_ripe_ip_map_score() -> None:
    for bgp_threshold in [20, 50, 10]:
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


def get_ripe_ip_map_schedule(
    targets: list[str],
    subnet_scores: dict,
    vps_per_subnet: dict,
    last_mile_delay: dict,
    cached_pings: dict,
) -> dict:
    """get all remaining measurments for ripe ip map evaluation"""
    target_schedule = {}
    for target in tqdm(targets):
        target_subnet = get_prefix_from_ip(target)
        target_scores = subnet_scores[target_subnet]

        if not target_scores:
            logger.error(f"{target_subnet} does not have score")
        for _, target_score in target_scores.items():

            # get vps, function of their subnet ecs score
            ecs_vps = get_ecs_vps(
                target_subnet,
                target_score,
                vps_per_subnet,
                last_mile_delay,
                probing_budget,
            )

            filtered_vps = []
            try:
                target_cached_pings = cached_pings[target]
                cached_vps = [vp_addr for vp_addr, _ in target_cached_pings]
                for vp_addr, score in ecs_vps:
                    if vp_addr in cached_vps:
                        continue

                    filtered_vps.append((vp_addr, score))

            except KeyError:
                filtered_vps = ecs_vps

            if filtered_vps:
                target_schedule[target] = [vp_addr for vp_addr, _ in filtered_vps]

    return target_schedule


def filter_targets(
    target_subnets: list[str],
    scores: dict,
    ping_vps_to_target: dict,
    latency_threshold: int,
    use_cache: bool = False,
    max_nb_targets: int = 5_000,
) -> list[str]:
    filtered_targets = []
    anycatch_db = load_anycatch_data()
    asndb = pyasn(str(path_settings.RIB_TABLE))

    geolocated_ip_addrs = []
    if use_cache:
        configs = []
        for file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
            if "ripe_ip_map" in file.name:
                config = load_json(file)
                configs.append((file.name, config))

        for file, config in configs:
            schedule = load_json(RIPEAtlasAPI().settings.MEASUREMENTS_SCHEDULE / file)
            geolocated_ip_addrs = [ip_addr for ip_addr, _ in schedule]
            logger.info(f"Nb addrs found in cache:: {len(geolocated_ip_addrs)}")

    anycast_ip_addr = 0
    filtere_ip_per_subent = defaultdict(list)
    for target in ping_vps_to_target:

        if target in geolocated_ip_addrs:
            continue

        target_subnet = get_prefix_from_ip(target)
        _, target_bgp_prefix = route_view_bgp_prefix(target, asndb)

        if target_subnet not in target_subnets:
            continue

        if not target_bgp_prefix:
            continue

        try:
            _ = scores[target_subnet]

            # filter anycast
            if target_bgp_prefix in anycatch_db:
                anycast_ip_addr += 1
                continue

            # filter per min rtt
            _, vp_min_rtt = min(ping_vps_to_target[target], key=lambda x: x[-1])

            if vp_min_rtt > latency_threshold:
                continue

            filtered_targets.append(target)

        except KeyError:
            continue

    for subnet, addrs in filtere_ip_per_subent.items():
        logger.debug(f"{subnet=}, {len(addrs)}")

    logger.info(f"IP anycast removed:: {anycast_ip_addr}")

    if len(filtered_targets) > max_nb_targets:
        shuffle(filtered_targets)
        filtered_targets = filtered_targets[:max_nb_targets]

    return filtered_targets


def get_measurement_schedule(score_file: Path) -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(PING_RIPE_IP_MAP_TABLE)
    cached_pings = get_pings_per_target(PING_RIPE_IP_MAP_GEO_RESOLVER)
    target_subnets = load_json(
        path_settings.END_TO_END_DATASET / "ripe_ip_map_subnets.json"
    )

    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    vps_per_subnet, _ = get_parsed_vps(vps, asndb, removed_vps)

    logger.info("BGP prefix score geoloc evaluation")

    scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)
    scores = scores.score_answer_subnets

    logger.info(f"Score retrieved for:: {len(scores)}")

    targets = filter_targets(
        target_subnets, scores, ping_vps_to_target, latency_threshold
    )

    logger.info(f"Ping schedule for:: {len(targets)}")

    measurement_schedule = get_ripe_ip_map_schedule(
        targets=targets,
        subnet_scores=scores,
        vps_per_subnet=vps_per_subnet,
        last_mile_delay=last_mile_delay,
        cached_pings=cached_pings,
    )

    logger.info(f"Ping schedule for:: {len(measurement_schedule)}")

    ping_schedule_with_id = []
    for target, selected_vps in measurement_schedule.items():
        # find vp id
        ping_schedule_with_id.append(
            (target, [vp_id_per_addr[vp_addr] for vp_addr in selected_vps])
        )

    return ping_schedule_with_id


async def ping_targets() -> None:
    """perfrom geolocation based on score similarity function"""
    score_path = path_settings.RESULTS_PATH / "ripe_ip_map_evaluation/"

    score_files = [
        score_path
        / "scores__best_hostname_geo_score_50_BGP_3_hostnames_per_org_ns.pickle",
        score_path
        / "scores__best_hostname_geo_score_10_BGP_3_hostnames_per_org_ns.pickle",
    ]

    for file in score_files:
        if "score_" in file.name:
            logger.info(f"Measurement config for file:: {file}")
            measurement_schedule = get_measurement_schedule(file)

            logger.info(f"Measurement schedule for {len(measurement_schedule)} targets")
            for i, (target, vp_ids) in enumerate(measurement_schedule):
                logger.debug(f"{target=}, {len(vp_ids)=}")
                if i > 10:
                    break

            await RIPEAtlasProber(
                probing_type="ping", probing_tag="ping_ripe_ip_map"
            ).main(measurement_schedule)


async def insert_measurements() -> None:
    cached_measurement_ids = await RIPEAtlasAPI().get_ping_measurement_ids(
        PING_RIPE_IP_MAP_GEO_RESOLVER
    )

    measurement_ids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_ripe_ip_map" in config_file.name:
            config = load_json(config_file)
            measurement_ids.extend([id for id in config["ids"]])

            logger.info(f"{config_file}:: {len(measurement_ids)} retrieved")

            measurement_to_insert = list(
                set(measurement_ids).difference(set(cached_measurement_ids))
            )

            logger.info(f"{len(measurement_to_insert)} measurements to insert")

    logger.info(f"Retreiving results for {len(measurement_to_insert)} measurements")

    await retrieve_pings(measurement_to_insert, PING_RIPE_IP_MAP_GEO_RESOLVER)


def load_ripe_ip_map_targets() -> None:
    pass


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


def evaluate(score_file: Path, latency_threhsold: int = 10_000) -> None:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    vps_per_subnet, _ = get_parsed_vps(vps, asndb, removed_vps)

    ping_vps_to_target = get_pings_per_target(
        PING_RIPE_IP_MAP_GEO_RESOLVER, removed_vps
    )
    targets = [target for target in ping_vps_to_target]

    scores: TargetScores = load_pickle(score_file)

    logger.info(f"ECS evaluation for score:: {score_file}, {len(targets)} targets")

    geo_resolver_shortest_pings = defaultdict(list)
    probing_budgets = [50]
    for probing_budget in probing_budgets:
        for target in tqdm(targets):
            target_subnet = get_prefix_from_ip(target)
            target_scores = scores.score_answer_subnets[target_subnet]

            # soi = detect_anycast(ping_vps_to_target[target], vp_distance_matrix)
            # if soi:
            #     logger.info(f"{target} is flagged as anycast")
            #     continue

            if not target_scores:
                logger.error(f"{target_subnet} does not have score")
            for _, target_score in target_scores.items():

                # get vps, function of their subnet ecs score
                ecs_vps = get_ecs_vps(
                    target_subnet,
                    target_score,
                    vps_per_subnet,
                    last_mile_delay,
                    probing_budget,
                )

                # find shortest ping in ecs vps
                selected_pings = []
                selected_vps_addr = [vp_addr for vp_addr, _ in ecs_vps]
                all_vps = ping_vps_to_target[target]
                for vp, min_rtt in all_vps:
                    if vp in selected_vps_addr:
                        selected_pings.append((vp, min_rtt))

                if not selected_pings:
                    continue

                shortest_ping_vp_addr, shortest_ping_rtt = min(
                    selected_pings, key=lambda x: x[-1]
                )
                geo_resolver_shortest_pings[probing_budget].append(
                    (target, shortest_ping_rtt)
                )

    ripe_ip_map_shortest_pings = get_ripe_ip_map_shortest_ping(targets)

    filtered_targets = set()
    filtered_ripe_ip_map_sp = []
    for target, min_rtt in ripe_ip_map_shortest_pings:
        if min_rtt < latency_threhsold:
            filtered_targets.add(target)
            filtered_ripe_ip_map_sp.append((target, min_rtt))

    filtered_geo_resolver_sp = defaultdict(list)
    for probing_budget, results in geo_resolver_shortest_pings.items():
        for target, min_rtt in results:
            if target in filtered_targets:
                filtered_geo_resolver_sp[probing_budget].append((target, min_rtt))

    logger.info(
        f"{len(filtered_geo_resolver_sp[50])} IP addresses remaining under {latency_threhsold}ms"
    )

    return filtered_geo_resolver_sp, filtered_ripe_ip_map_sp


async def main() -> None:
    retrieve_public_measurements = False
    ecs_resoltion = False
    calculate_score = False
    geolocate = False
    insert = False
    evaluation = True

    if retrieve_public_measurements:
        await insert_results_from_tag()

    if ecs_resoltion:
        await resolve_subnets()

    if calculate_score:
        get_ripe_ip_map_score()

    if geolocate:
        await ping_targets()

    if insert:
        await insert_measurements()

    if evaluation:
        score_files = [
            path_settings.RESULTS_PATH
            / "ripe_ip_map_evaluation/scores__best_hostname_geo_score_10_BGP_3_hostnames_per_org_ns.pickle",
            path_settings.RESULTS_PATH
            / "ripe_ip_map_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle",
            path_settings.RESULTS_PATH
            / "ripe_ip_map_evaluation/scores__best_hostname_geo_score_50_BGP_3_hostnames_per_org_ns.pickle",
        ]
        geo_resolver_sp = {}
        for score_file in score_files:
            bgp_prefix_threshold = score_file.name.split("score_")[-1].split("_")[0]
            nb_hostnames_per_org_ns = score_file.name.split("BGP_")[-1].split("_")[0]
            logger.info(f"{score_file}=")
            (
                geo_resolver_sp[(bgp_prefix_threshold, nb_hostnames_per_org_ns)],
                ripe_ip_map_sp,
            ) = evaluate(score_file)

        plot_ripe_ip_map(
            geo_resolver_sp,
            ripe_ip_map_sp,
            output_path="ripe_ip_map_evaluation",
        )


if __name__ == "__main__":
    asyncio.run(main())
