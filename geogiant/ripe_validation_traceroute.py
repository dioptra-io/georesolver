import asyncio
import time

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from loguru import logger
from pych_client import ClickHouseClient

from geogiant.clickhouse import InsertFromCSV, CreateTracerouteTable
from geogiant.common.files_utils import load_json, dump_json, create_tmp_csv_file
from geogiant.common.geoloc import distance
from geogiant.common.queries import load_vps
from geogiant.prober import RIPEAtlasProber, RIPEAtlasAPI
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def compute_vp_distance_matrix(vps: list[dict]) -> None:
    """
    calculate distance from one VP to all the others,
    only keep the first thousands closest VPs
    """
    logger.info("Computing VP distance matrix")

    vp_pairwise_distance = defaultdict(list)
    for vp_i in tqdm(vps):
        for vp_j in vps:
            if vp_i["addr"] == vp_j["addr"]:
                continue

            d = distance(vp_i["lat"], vp_j["lat"], vp_i["lon"], vp_j["lon"])
            vp_pairwise_distance[vp_i["addr"]].append((vp_j["addr"], d))

        vp_pairwise_distance[vp_i["addr"]] = sorted(
            vp_pairwise_distance[vp_i["addr"]], key=lambda x: x[-1]
        )
        vp_pairwise_distance[vp_i["addr"]] = vp_pairwise_distance[vp_i["addr"]][:50]

    dump_json(
        data=vp_pairwise_distance, output_file=path_settings.VPS_PAIRWISE_DISTANCE
    )


def get_measurement_schedule(dry_run: bool = False) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    max_nb_traceroute = 50
    vps = load_vps(clickhouse_settings.VPS_FILTERED)

    if dry_run:
        logger.debug(f"Dry run:: {len(vps)} VPs")

    if not path_settings.VPS_PAIRWISE_DISTANCE.exists():
        logger.debug(
            f"VP distance matrix file:: {path_settings.VPS_PAIRWISE_DISTANCE} does not exists, calculating it"
        )
        compute_vp_distance_matrix(vps)

    vps_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

    logger.debug(f"{len(vps_distance_matrix)=}")

    # get vp id per addr
    vp_addr_to_id = {}
    for vp in vps:
        vp_addr_to_id[vp["addr"]] = vp["id"]

    # parse distance matrix
    ordered_distance_matrix = {}
    for vp in vps:
        distances = vps_distance_matrix[vp["addr"]]
        distances = sorted(distances.items(), key=lambda x: x[-1])
        ordered_distance_matrix[vp["addr"]] = distances[:max_nb_traceroute]

    traceroute_targets_per_vp = {}
    for vp in vps:
        closest_vps = ordered_distance_matrix[vp["addr"]][:max_nb_traceroute]

        closest_vp_ids = []
        for vp_addr, _ in closest_vps:
            try:
                closest_vp_ids.append((vp_addr, vp_addr_to_id[vp_addr]))
            except KeyError:
                continue

        traceroute_targets_per_vp[vp["id"]] = closest_vp_ids

    logger.debug(f"{len(traceroute_targets_per_vp)=}")

    # group VPs that must trace the same target to maximize parallel measurements
    traceroute_schedule = defaultdict(set)
    for vp_id, closest_vp_ids in traceroute_targets_per_vp.items():
        for addr, id in closest_vp_ids:
            traceroute_schedule[addr].add(vp_id)

    logger.info(
        f"Traceroute Schedule for:: {len(traceroute_schedule.values())} targets"
    )
    count = 0
    for addr, ids in traceroute_schedule.items():
        count += len(ids)
    logger.info(f"Total number of traceroute:: {count}")

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    measurement_schedule = []
    for target, vps in traceroute_schedule.items():
        vps = list(vps)
        for i in range(0, len(vps), 1):
            batch_vps = vps[i * batch_size : (i + 1) * batch_size]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target,
                    [id for id in batch_vps],
                )
            )

    return measurement_schedule


def get_latest_measurement_config() -> list[int]:
    """return all measurement ids from latest traceroute measurement config"""
    timestamp = None
    latest_config = None

    for file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "traceroute" in file.name:
            logger.debug(f"found traceroute config:: {file.name}")

            new_config = load_json(file)
            new_timestamp = new_config["start_time"].split(".")[0]
            new_timestamp = datetime.strptime(new_timestamp, "%Y-%m-%d %H:%M:%S")

            if not timestamp:
                timestamp = new_timestamp
                latest_config = new_config

            if new_timestamp >= timestamp:
                latest_config = new_config
                timestamp = new_timestamp

    if latest_config:
        return latest_config


def get_all_measurement_ids() -> list[int]:
    """retrive all traceroute measurement ids from config file"""
    ids = set()
    for file in RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG.iterdir():
        if "traceroute" in file.name:
            logger.info(f"Found traceroute measurement config:: {file.name}")
            config = load_json(file)
            ids.update(config["ids"])

    return ids


async def retrieve_all_traceroutes(ids: int) -> list[dict]:
    """retrieve all traceroutes from a list of ids"""
    traceroute_results = []
    for id in tqdm(ids):
        traceroute_result = await RIPEAtlasAPI().get_traceroute_results(id)
        traceroute_results.extend(traceroute_result)

        time.sleep(0.1)

    return traceroute_results


async def main() -> None:
    make_measurement = False
    insert_measurement = True

    if make_measurement:
        traceroute_config = get_latest_measurement_config()
        latest_schedule = load_json(
            RIPEAtlasAPI().settings.MEASUREMENTS_SCHEDULE
            / f"traceroute__{traceroute_config['uuid']}.json"
        )

        schedule: list = deepcopy(latest_schedule)
        if traceroute_config:
            for id in tqdm(traceroute_config["ids"]):
                traceroute_info = await RIPEAtlasAPI().get_traceroute_info(id)

                # find corresponding measurement in schedule
                found_traceroute = False
                for i, (target, probe_requested) in enumerate(latest_schedule):
                    if (
                        target == traceroute_info["target"]
                        and len(probe_requested) == traceroute_info["probes_requested"]
                    ):
                        schedule.remove([target, probe_requested])
                        found_traceroute = True

                if not found_traceroute:
                    raise RuntimeError(
                        "Traceroute does not exists in original schedule"
                    )

                time.sleep(0.1)

            logger.info(
                f"Nb targets:: original schedule:: {len(latest_schedule)}, updated schedule:: {len(schedule)}"
            )
        else:
            logger.debug("No preivous measurements were done")
            schedule = get_measurement_schedule(dry_run=False)

        await RIPEAtlasProber("traceroute").main(schedule)

    if insert_measurement:
        traceroute_ids = get_all_measurement_ids()

        logger.info(f"Retrieved {len(traceroute_ids)} traceroute measurements")

        traceroute_results = await retrieve_all_traceroutes(traceroute_ids)

        tmp_file_path = create_tmp_csv_file(traceroute_results)

        with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
            CreateTracerouteTable().execute(
                client, clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
            )
            InsertFromCSV().execute_from_in_file(
                table_name=clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY,
                in_file=tmp_file_path,
            )

        tmp_file_path.unlink()


if __name__ == "__main__":
    logger.info("Starting validation Ping measurement on all RIPE atlas anchors")
    asyncio.run(main())
