import asyncio
import httpx
import time

from dateutil import parser
from pathlib import Path
from datetime import datetime, timedelta
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pych_client import ClickHouseClient


from geogiant.clickhouse import GetDstPrefix
from geogiant.prober.ripe_api import RIPEAtlasAPI
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.evaluation.ecs_geoloc_eval import get_ecs_vps
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.queries import get_min_rtt_per_vp, get_pings_per_target, load_vps
from geogiant.common.utils import TargetScores, get_parsed_vps
from geogiant.common.files_utils import (
    load_csv,
    load_json,
    dump_json,
    load_pickle,
    load_anycatch_data,
    decompress,
    load_iter,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# CLICKHOUSE TABLE
ping_ripe_altas_table = "pings_ripe_altas"
ping_ripe_ip_map_table = "pings_ripe_ip_map"
ecs_table = "ripe_ip_map_mapping_ecs"
vps_ecs_table = "vps_mapping_ecs"

# FILE PATHS
ripe_ip_map_subnets_path = path_settings.DATASET / "ripe_ip_map_subnets.json"
measurement_ids_path = path_settings.DATASET / "ripe_ip_map_ids.json"
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
latency_threshold = 4
probing_budget = 500

START_DATE: str = "2024-04-16"
END_DATE: str = "2024-04-17"


def download(
    file_url: Path,
    output_file_name: str,
    output_dir: Path,
) -> None:
    """download public measurement from RIPE Atlas FTP, output results into clickhouse"""

    output_file = output_dir / output_file_name

    if not output_file.parent.exists():
        output_file.parent.mkdir()

    try:
        with output_file.open("wb") as f:
            with httpx.stream("GET", file_url) as resp:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        logger.info(f"File successfully downloaded")
    except httpx.RemoteProtocolError:
        logger.info(f"Peer closed connection, could not download file {file_url}")


def download_pings(
    output_dir: Path = path_settings.RIPE_ATLAS_PUBLIC_PINGS,
) -> None:
    """download all traceroutes archive between a starting and ending date"""
    base_url = "https://data-store.ripe.net/datasets/atlas-daily-dumps/"

    start_date = parser.parse(START_DATE)
    end_date = parser.parse(END_DATE)

    while start_date <= end_date:
        daily_archive_url = base_url + str(start_date.date()) + "/"
        # download all dumps from a day worth of data
        for i in range(0, 24, 1):
            file_name = (
                "ping-"
                + str(start_date.date())
                + f"T{'0'+str(i) if i < 10 else str(i)}00.bz2"
            )

            file_url = daily_archive_url + file_name

            logger.info(f"Downloading file:: {file_url}")

            download(file_url, file_name, output_dir)

        start_date = start_date + timedelta(days=1)


def get_ripe_ip_map_ids(tag: str = "single-radius") -> None:
    """get all measurements from RIPE IP map single radius engine"""
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_time = datetime.strptime(END_DATE, "%Y-%m-%d")

    start_time = int(start_time.timestamp())
    end_time = int(end_time.timestamp())

    print(str(datetime.fromtimestamp(start_time, tz=None)))
    print(str(datetime.fromtimestamp(end_time, tz=None)))

    logger.info(f"Fetching measurement metadata from:: {START_DATE} to {END_DATE}")

    params = {
        "tags": tag,
        "type": "ping",
        "af": 4,
        "start_time__gte": start_time,
        "stop_time__lte": end_time,
    }
    return RIPEAtlasAPI().get_tag_measurement_ids(params=params)


async def insert_public_pings(file_path: Path) -> None:
    ping_results = []
    batch_size = 100_000

    if "meta" in file_path.name:
        return

    # assume that not all file was decompressed
    if ".bz2" in file_path.name:
        logger.info(f"Decompressing file:: {file_path}")
        decompressed_file_path = decompress(file_path)
        file_path.unlink()
        logger.info(f"Removed file:: {file_path}")
        file_path = decompressed_file_path
    else:
        logger.info(f"inserting:: {file_path} in clickhouse DB")

    for ping in load_iter(file_path):
        if not ping:
            continue

        if ping["af"] != 4 or not ping["from"]:
            continue

        ping_result = RIPEAtlasAPI().parse_ping([ping])
        ping_results.extend(ping_result)

        if len(ping_results) > batch_size:

            logger.info(f"inserting batch of traceroute: limit = {batch_size}")
            await RIPEAtlasAPI().insert_pings(
                ping_results, table_name=ping_ripe_altas_table
            )
            ping_results = []

    # file_path.unlink()
    logger.info(f"Removed file:: {file_path}")


async def get_ripe_ip_map_from_ids(use_cache: bool = True) -> None:
    ping_results = []
    batch_size = 100

    measurement_ids = load_json(measurement_ids_path)
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

    # filter out ids that were flagged without results
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
            time.sleep(0.5)

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


async def get_ripe_ip_map_results() -> None:

    previous_measurement_ids = load_json(measurement_ids_path)
    measurement_ids = get_ripe_ip_map_ids()
    measurement_ids = list(set(previous_measurement_ids).union(set(measurement_ids)))

    # load and dump
    dump_json(measurement_ids, measurement_ids_path)

    if not path_settings.RIPE_ATLAS_PUBLIC_PINGS.exists():
        logger.info("Downloading ping files from RIPE Atlas FTP server")
        download_pings()

    for file_path in path_settings.RIPE_ATLAS_PUBLIC_PINGS.iterdir():
        await insert_public_pings(
            file_path=file_path,
        )


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


def get_subnets() -> list[str]:
    ripe_ip_map_subnets = get_ripe_ip_map_subnets(latency_threshold)

    logger.info(
        f"Retrieved:: {len(ripe_ip_map_subnets)} RIPE IP map subnets, {latency_threshold=}"
    )
    dump_json(ripe_ip_map_subnets, ripe_ip_map_subnets_path)


async def resolve_subnets() -> None:
    selected_hostnames = load_csv(
        path_settings.DATASET / "selected_hostname_geo_score.csv"
    )

    get_subnets()

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=ripe_ip_map_subnets_path,
        output_table=ecs_table,
        chunk_size=100,
    )


def get_ripe_ip_map_score() -> None:
    # ripe_ip_map_subnets_path.unlink()
    get_subnets()

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
        "targets_subnet_path": ripe_ip_map_subnets_path,
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
    count = 0
    for target in tqdm(targets):
        target_subnet = get_prefix_from_ip(target)
        target_scores = subnet_scores[target_subnet]

        if not target_scores:
            logger.error(f"{target_subnet} does not have score")
        for _, target_score in target_scores.items():

            # get vps, function of their subnet ecs score
            ecs_vps = get_ecs_vps(
                target_subnet, target_score, vps_per_subnet, last_mile_delay, 5_00
            )

            count += 1

            # remove vps that have a high last mile delay
            # ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

            # ecs_vps = select_one_vp_per_as_city(
            #     ecs_vps, vps_coordinates, last_mile_delay
            # )[:probing_budget]

            target_schedule[target] = [vp_addr for vp_addr, _ in ecs_vps]

    print(count)

    return target_schedule


def filter_targets(
    targets: list[str],
    scores: dict,
    ping_vps_to_target: dict,
    latency_threshold: int,
) -> list[str]:
    filtered_targets = []
    anycatch_db = load_anycatch_data()
    asndb = pyasn(str(path_settings.RIB_TABLE))

    anycast_ip_addr = 0
    filtere_ip_per_subent = defaultdict(list)
    for target in targets:
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

    return filtered_targets


def get_measurement_schedule() -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(ping_ripe_ip_map_table, removed_vps)
    targets = set([target for target in ping_vps_to_target.keys()])

    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    vps_per_subnet, _ = get_parsed_vps(vps, asndb)

    logger.info("BGP prefix score geoloc evaluation")

    scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)
    scores = scores.score_answer_bgp_prefixes

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

    ping_schedule_with_id = defaultdict(list)
    for target, selected_vps in measurement_schedule.items():
        for vp_addr in selected_vps:

            # find vp id
            ping_schedule_with_id[target].append(vp_id_per_addr[vp_addr])

    return measurement_schedule


def ping_targets() -> None:
    """perfrom geolocation based on score similarity function"""
    ping_schedule = get_measurement_schedule()

    # for target, vp_ids in ping_schedule.items():
    #     logger.debug(f"{target=}, {len(vp_ids)} VPs")


async def main() -> None:
    retrieve_public_measurements = True
    ecs_resoltion = False
    calculate_score = False
    geolocate = False

    if retrieve_public_measurements:
        await get_ripe_ip_map_from_ids()

    if ecs_resoltion:
        await resolve_subnets()

    if calculate_score:
        get_ripe_ip_map_score()

    if geolocate:
        ping_targets()


if __name__ == "__main__":
    asyncio.run(main())
