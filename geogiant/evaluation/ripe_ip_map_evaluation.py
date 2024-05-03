import asyncio
import time


from ipaddress import IPv4Address
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
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    ecs_dns_vp_selection_eval,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    get_pings_per_target,
    load_vps,
    load_target_subnets,
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
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# CLICKHOUSE TABLE
ping_ripe_altas_table = "pings_ripe_altas"
ping_ripe_ip_map_table = "pings_ripe_ip_map"
ping_ripe_ip_map_dioptra = "ping_ripe_ip_map_dioptra"
ecs_table = "ripe_ip_map_evaluation_mapping_ecs"
vps_ecs_table = "vps_mapping_ecs"

# FILE PATHS
ripe_ip_map_subnets_path = path_settings.DATASET / "ripe_ip_map_subnets.json"
measurement_ids_path = path_settings.DATASET / "ripe_ip_map_ids.json"
best_measurement_ids_path = path_settings.DATASET / "ripe_ip_map_locate_ids.json"
filtered_measurement_ids_path = path_settings.DATASET / "ripe_ip_map_filtered_ids.json"
vps_subnet_path = path_settings.DATASET / "vps_subnet.json"
score_file = (
    path_settings.RESULTS_PATH
    / f"ripe_ip_map_evaluation/scores__best_hostname_geo_score.pickle"
)
results_file = (
    path_settings.RESULTS_PATH
    / f"ripe_ip_map_evaluation/{'results' + str(score_file).split('scores')[-1]}"
)

# CONSTANT PARAMETERS
latency_threshold = 10
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
        params=params, ping_table=ping_ripe_ip_map_table
    )


def get_ids_from_ripe_ip_map() -> None:

    # download data
    measurement_ids = load_json(best_measurement_ids_path)
    old_targets = []
    for id in measurement_ids:
        try:
            IPv4Address(id)
            old_targets.append(id)
        except ValueError:
            continue

    pings_router_2ms = get_pings_per_target("pings_routers_2ms")

    if not (path_settings.DATASET / "ripe_ip_map_targets.json").exists():
        targets = [target for target in pings_router_2ms]
        shuffle(targets)
        targets = targets[: max_nb_measurements + len(old_targets)]

        dump_json(targets, path_settings.DATASET / "ripe_ip_map_targets.json")

    targets = load_json(path_settings.DATASET / "ripe_ip_map_targets.json")

    filtered_targets = []
    for target in targets:
        if target in old_targets:
            continue

        filtered_targets.append(target)

    logger.info(f"Starting measurements on:: {len(filtered_targets)}")

    for target in filtered_targets:
        # url
        measurement_id = RIPEAtlasAPI().ripe_ip_map_locate(target)
        if measurement_id:
            measurement_ids.append(measurement_id)
            measurement_ids.append(target)
        else:
            measurement_ids.append(target)
            time.sleep(5)

        logger.info(f"{target=}, {measurement_id=}")
        time.sleep(1)

        if len(measurement_ids) > 10:
            old_measurement_ids = load_json(best_measurement_ids_path)
            measurement_ids = list(set(measurement_ids).union(set(old_measurement_ids)))
            dump_json(measurement_ids, best_measurement_ids_path)

    old_measurement_ids = load_json(best_measurement_ids_path)
    measurement_ids = list(set(measurement_ids).union(set(old_measurement_ids)))
    dump_json(measurement_ids, best_measurement_ids_path)


def get_ripe_ip_map_ids() -> None:
    """get all measurements from RIPE IP map single radius engine"""
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d")
    start_time = int(start_time.timestamp())

    ripe_ip_map_ids = load_json(best_measurement_ids_path)
    logger.info(f"Retrieving measurements for:: {len(ripe_ip_map_ids)}")

    parsed_ripe_ip_map_ids = []
    for id in ripe_ip_map_ids:
        try:
            IPv4Address(id)
            logger.info(
                f"Fetching measurement metadata from:: {START_DATE} to {END_DATE}"
            )

            params = {
                "tags": "single-radius",
                "target_ip": id,
                "type": "ping",
                "af": 4,
                "start_time__gte": start_time,
            }
            msm_id = RIPEAtlasAPI().get_tag_measurement_ids(params=params)
        except ValueError:
            msm_id = [id]

        parsed_ripe_ip_map_ids.extend(msm_id)

    dump_json(
        parsed_ripe_ip_map_ids, path_settings.DATASET / "parsed_ripe_ip_map_ids.json"
    )


async def get_ripe_ip_map_from_ids(use_cache: bool = False) -> None:
    ping_results = []
    batch_size = 100

    measurement_ids = load_json(path_settings.DATASET / "parsed_ripe_ip_map_ids.json")
    remaining_ids = set()
    if use_cache:
        cached_measurement_ids = await RIPEAtlasAPI().get_ping_measurement_ids(
            ping_ripe_ip_map_table
        )
        remaining_ids = set(measurement_ids).symmetric_difference(
            set(cached_measurement_ids)
        )

    if remaining_ids:
        measurement_ids = remaining_ids

    # # filter out ids that were flagged without results
    filetered_ids = set()
    if (filtered_measurement_ids_path).exists():
        filetered_ids = load_json(filtered_measurement_ids_path)
        filetered_ids = set(filetered_ids)

    if filetered_ids:
        measurement_ids = set(filetered_ids).symmetric_difference(set(measurement_ids))

    for i, measurement_id in enumerate(measurement_ids):
        logger.info(
            f"Retreiving measurement uuid:: {measurement_id} ({i+1}/{len(measurement_ids)})"
        )
        ping_result = RIPEAtlasAPI().get_ping_results(measurement_id)

        logger.debug(f"Nb of pings retreived:: {len(ping_result)}")

        if len(ping_result) < 1:
            filetered_ids.add(measurement_id)
        else:
            time.sleep(1)

        ping_results.extend(ping_result)

        if len(ping_results) > batch_size:
            await RIPEAtlasAPI().insert_pings(
                ping_results, table_name=ping_ripe_ip_map_table
            )
            ping_results = []
            old_filtered_ids = []
            if filtered_measurement_ids_path.exists():
                old_filtered_ids = load_json(filtered_measurement_ids_path)

            filetered_ids.update(old_filtered_ids)
            dump_json(
                list(filetered_ids),
                filtered_measurement_ids_path,
            )

    await RIPEAtlasAPI().insert_pings(ping_results, table_name=ping_ripe_ip_map_table)


async def get_ripe_ip_map_results(get_measurement_ids: bool = False) -> None:

    if get_measurement_ids:
        previous_measurement_ids = load_json(measurement_ids_path)
        measurement_ids = get_ripe_ip_map_ids()
        measurement_ids = list(
            set(previous_measurement_ids).union(set(measurement_ids))
        )

        # load and dump
        dump_json(measurement_ids, measurement_ids_path)

    await get_ripe_ip_map_from_ids()


def get_ripe_ip_map_subnets(latency_threshold: int) -> list:
    """retrieve all RIPE IP map subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetDstPrefix().execute(
            client=client,
            table_name=ping_ripe_ip_map_table,
            latency_threshold=latency_threshold,
        )

    ripe_ip_map_subnets = []
    for row in rows:
        ripe_ip_map_subnets.append(row["dst_prefix"])

    return ripe_ip_map_subnets


def get_subnets_from_pings(use_cache: bool = False) -> list[str]:
    ripe_ip_map_subnets = get_ripe_ip_map_subnets(latency_threshold)

    if use_cache:
        resolved_subnets = load_target_subnets(ecs_table)
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
    selected_hostnames = set()
    for bgp_threshold in [5, 10, 20, 50, 100]:
        selected_hostnames_per_cdn = load_json(
            path_settings.DATASET
            / f"hostname_geo_score_selection_{bgp_threshold}_BGP.json"
        )

        for org, hostnames in selected_hostnames_per_cdn.items():
            selected_hostnames.update(hostnames)

    logger.info(f"Nb total hostnames:: {len(selected_hostnames)}")
    dump_csv(
        selected_hostnames,
        path_settings.DATASET / "all_threshold_geo_score_hostnames.csv",
    )

    selected_hostnames = load_csv(
        path_settings.DATASET / "all_threshold_geo_score_hostnames.csv",
    )

    get_subnets_from_pings()

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=ripe_ip_map_subnets_path,
        output_table=ecs_table,
        chunk_size=100,
    )


def get_ripe_ip_map_score() -> None:
    selected_hostnames_per_cdn = load_json(
        path_settings.DATASET / "hostname_geo_score_selection.json"
    )

    selected_hostnames = set()
    for org, hostnames in selected_hostnames_per_cdn.items():
        logger.info(f"{org=}, {len(hostnames)=}")
        selected_hostnames.update(hostnames)

    logger.info(f"{len(selected_hostnames)=}")

    for org, hostnames in selected_hostnames_per_cdn.items():
        logger.info(f"{org=}, {len(hostnames)=}")

    score_config = {
        "targets_table": ecs_table,
        "vps_subnet_path": vps_subnet_path,
        "hostname_per_cdn": selected_hostnames_per_cdn,
        "selected_hostnames": selected_hostnames,
        "targets_ecs_table": ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": [
            "jaccard",
        ],
        "answer_granularities": [
            "answer_subnets",
        ],
        "output_path": score_file,
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

            target_schedule[target] = [vp_addr for vp_addr, _ in ecs_vps]

    return target_schedule


def filter_targets(
    targets: list[str],
    scores: dict,
    ping_vps_to_target: dict,
    latency_threshold: int,
    use_cache: bool = True,
    max_nb_targets: int = 1500,
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
    for target in targets:

        if target in geolocated_ip_addrs:
            continue

        target_subnet = get_prefix_from_ip(target)
        _, target_bgp_prefix = route_view_bgp_prefix(target, asndb)

        if not target_bgp_prefix:
            continue

        try:
            _ = scores[target_subnet]

            # filter per min rtt
            _, vp_min_rtt = min(ping_vps_to_target[target], key=lambda x: x[-1])

            if vp_min_rtt > latency_threshold:
                continue

            # filter anycast
            if target_bgp_prefix in anycatch_db:
                anycast_ip_addr += 1
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


def get_measurement_schedule() -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(ping_ripe_ip_map_table)
    targets = set([target for target in ping_vps_to_target.keys()])

    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    vps_per_subnet, _ = get_parsed_vps(vps, asndb, removed_vps)

    logger.info("BGP prefix score geoloc evaluation")

    scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)
    scores = scores.score_answer_subnets

    logger.info(f"Score retrieved for:: {len(scores)}")

    targets = filter_targets(targets, scores, ping_vps_to_target, latency_threshold)

    logger.info(f"Ping schedule for:: {len(targets)}")

    measurement_schedule = get_ripe_ip_map_schedule(
        targets=targets,
        subnet_scores=scores,
        vps_per_subnet=vps_per_subnet,
        last_mile_delay=last_mile_delay,
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
    measurement_schedule = get_measurement_schedule()

    await RIPEAtlasProber(probing_type="ping", probing_tag="ping_ripe_ip_map").main(
        measurement_schedule
    )


async def insert_measurements() -> None:
    # retrive ping measurements and insert them into clickhouse db
    measurement_ids = []
    for config_file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "ping_ripe_ip_map" in config_file.name:
            config = load_json(config_file)
            measurement_ids.extend([id for id in config["ids"]])

    logger.info(f"Retreiving results for {len(measurement_ids)} measurements")

    await retrieve_pings(measurement_ids, ping_ripe_ip_map_dioptra)


def load_ripe_ip_map_targets() -> None:
    pass


def get_ripe_ip_map_shortest_ping(targets: list[str]) -> None:
    """for each IP addresses retrieved from RIPE IP map, get the shortest ping"""
    pings_single_radius = get_pings_per_target(ping_ripe_ip_map_table)

    shortest_ping_per_target = []
    for target, pings in pings_single_radius.items():
        if target in targets:
            _, shortest_ping_rtt = min(pings, key=lambda x: x[-1])
            shortest_ping_per_target.append((target, shortest_ping_rtt))

    return shortest_ping_per_target


def evaluate() -> None:
    """calculate distance error and latency for each score"""
    probing_budgets = [10, 20, 50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    ping_vps_to_target = get_pings_per_target(ping_ripe_ip_map_dioptra, removed_vps)
    targets = [target for target in ping_vps_to_target]

    scores: TargetScores = load_pickle(score_file)

    logger.info(f"ECS evaluation for score:: {score_file}, {len(targets)} targets")

    geo_resolver_shortest_pings = []
    for target in tqdm(targets):
        target_subnet = get_prefix_from_ip(target)
        target_scores = scores.score_answer_subnets[target_subnet]

        if not target_scores:
            logger.error(f"{target_subnet} does not have score")
        for _, target_score in target_scores.items():

            # get vps, function of their subnet ecs score
            ecs_vps = get_ecs_vps(
                target_subnet,
                target_score,
                vps_per_subnet,
                last_mile_delay,
                50,
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
            geo_resolver_shortest_pings.append((target, shortest_ping_rtt))

    ripe_ip_map_shortest_pings = get_ripe_ip_map_shortest_ping(targets)

    filtered_targets = set()
    filtered_ripe_ip_map_sp = []
    for target, min_rtt in ripe_ip_map_shortest_pings:
        if min_rtt < 10:
            filtered_targets.add(target)
            filtered_ripe_ip_map_sp.append((target, min_rtt))

    filtered_geo_resolver_sp = []
    for target, min_rtt in geo_resolver_shortest_pings:
        if target in filtered_targets:
            filtered_geo_resolver_sp.append((target, min_rtt))

    logger.info(f"{len(filtered_geo_resolver_sp)} IP addresses remaining under 10ms")

    return filtered_geo_resolver_sp, filtered_ripe_ip_map_sp


async def main() -> None:
    retrieve_public_measurements = False
    ecs_resoltion = True
    calculate_score = False
    geolocate = False
    insert = False
    evaluation = False

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
        geo_resolver_sp, ripe_ip_map_sp = evaluate()
        plot_ripe_ip_map(
            geo_resolver_sp, ripe_ip_map_sp, output_path="ripe_ip_map_evaluation"
        )


if __name__ == "__main__":
    asyncio.run(main())
