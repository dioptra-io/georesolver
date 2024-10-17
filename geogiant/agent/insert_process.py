import asyncio

from tqdm import tqdm
from pyasn import pyasn
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta

from geogiant.prober import RIPEAtlasAPI
from geogiant.common.queries import (
    load_cached_targets,
    get_measurement_ids,
    insert_geoloc,
    load_target_geoloc,
    load_vps,
    insert_pings,
    insert_traceroutes,
)
from geogiant.common.files_utils import load_csv
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def watch_measurements(
    uuid: str,
    measurement_ids: list[int],
    ongoing_measurements: list[int],
    measurement_timeout: int = 60 * 15,
) -> list[int]:
    """check if ongoing measurements are running for longer than defined timeout"""
    measurement_stopped = []
    current_time = datetime.now()
    for id, start_time in measurement_ids:
        if not id in ongoing_measurements:
            continue

        start_time = datetime.fromtimestamp(start_time)
        if start_time + timedelta(seconds=measurement_timeout) > current_time:
            logger.info(f"Measurement {uuid}:: measurement {id} timed out")
            stopped_id = RIPEAtlasAPI().stop_measurement(id)
            measurement_stopped.append(stopped_id)

    return measurement_stopped


def parse_geoloc_data(target_geoloc: dict) -> list[str]:
    """parse ping data to geoloc csv"""
    csv_data = []
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_RAW_TABLE)

    vps_per_vps_addr = {}
    for vp in vps:
        vps_per_vps_addr[vp["addr"]] = vp

    for target_addr, shortest_ping_data in target_geoloc.items():
        target_subnet = get_prefix_from_ip(target_addr)
        target_asn, target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)
        if not target_bgp_prefix or not target_asn:
            target_bgp_prefix = "Unknown"
            target_asn = -1

        msm_id = shortest_ping_data[1]
        min_rtt = shortest_ping_data[2]

        # filter unresponisive targets
        if min_rtt == -1:
            continue

        vp_addr = shortest_ping_data[0]
        try:
            vp = vps_per_vps_addr[vp_addr]
        except KeyError:
            logger.debug(f"VP {vp_addr} does not exists")
            continue

        csv_data.append(
            f"{target_addr},\
            {target_subnet},\
            {target_bgp_prefix},\
            {target_asn},\
            {vp['lat']},\
            {vp['lon']},\
            {vp['country_code']},\
            {vp_addr},\
            {vp['subnet']},\
            {vp['bgp_prefix']},\
            {vp['asn_v4']},\
            {min_rtt},\
            {msm_id}"
        )

    return csv_data


async def insert_geoloc_from_pings(ping_table: str, geoloc_table: str) -> None:
    """insert all geoloc in clickhouse"""
    target_geoloc = load_target_geoloc(table_name=ping_table)
    cached_geoloc_msm_ids = get_measurement_ids(geoloc_table)

    filtered_geoloc = {}
    for target, geoloc in target_geoloc.items():
        msm_id = geoloc[1]
        if msm_id in cached_geoloc_msm_ids:
            continue

        filtered_geoloc[target] = geoloc

    csv_data = parse_geoloc_data(filtered_geoloc)

    await insert_geoloc(
        csv_data=csv_data,
        output_table=geoloc_table,
    )


async def retrieve_pings(
    ids: list[int],
    output_table: str,
    wait_time: int = 0.1,
    output_logs: Path = None,
) -> None:
    """retrieve all ping measurements from a list of measurement ids"""
    csv_data = []

    if output_logs:
        output_file = output_logs.open("a")
    else:
        output_file = None

    for id in tqdm(ids, file=output_file):
        ping_results = await RIPEAtlasAPI().get_ping_results(id)
        csv_data.extend(ping_results)

        await asyncio.sleep(wait_time)

    if output_file:
        output_file.close()

    await insert_pings(csv_data, output_table)


async def retrieve_traceroutes(
    ids: list[int],
    output_table: str,
    wait_time: float = 0.1,
    output_logs: Path = None,
) -> list[dict]:
    """retrieve all traceroutes from a list of ids"""
    if output_logs:
        output_file = output_logs.open("a")
    else:
        output_file = None

    csv_data = []
    for id in tqdm(ids):
        traceroute_result = await RIPEAtlasAPI().get_traceroute_results(id)
        csv_data.extend(traceroute_result)

        await asyncio.sleep(wait_time)

    if output_file:
        output_file.close()

    await insert_traceroutes(csv_data, output_table)


def filter_targets(
    targets: list[str],
    geolocated_targets: list[str],
    score_table: str,
    ping_table: str,
    verbose: bool = False,
) -> list[str]:
    """return targets for which pings were not made"""
    filtered_targets = []

    # get cached target from table
    cached_targets = load_cached_targets(ping_table)

    # add targets that were geolocated but not inserted
    cached_targets.extend(geolocated_targets)
    # get remaining targets to geolocate
    no_measured_target = set(targets).difference(set(cached_targets))

    # remove targets for which we do not have scores ready
    for target in no_measured_target:
        target_subnet = get_prefix_from_ip(target)
        if target_subnet in cached_score_subnets:
            filtered_targets.append(target)

    # remove targets for which a measurement was started but results not inserted yet
    filtered_targets = set(filtered_targets).difference(set(geolocated_targets))

    return filtered_targets, no_measured_target


async def insert_results(
    probing_type: str,
    probing_tag: str,
    measurement_table: str,
    geoloc_table: str,
    nb_targets: int,
    wait_time: int = 60,
    output_logs: Path = None,
    batch_size: int = 1_000,
) -> None:
    """insert ongoing measurements"""
    cached_inserted_measurements = get_measurement_ids(measurement_table)
    current_time = datetime.timestamp(datetime.now() - timedelta(days=2))

    insert_done = False
    while not insert_done:

        # load previously cached targets
        cached_targets = load_cached_targets(targets)
        # get targets that still have to be inserted
        remaining_targets = set(targets).difference(set(cached_targets))

        # Get all stopped measurements
        stopped_measurement_ids = await RIPEAtlasAPI().get_stopped_measurement_ids(
            start_time=current_time,
            tags=["dioptra", probing_tag],
        )

        # Find stopped measurements that are not inserted yet
        measurement_to_insert = set(stopped_measurement_ids).difference(
            set(cached_inserted_measurements)
        )

        if not measurement_to_insert:
            logger.info(f"{probing_tag} :: No measurement to insert")
            await asyncio.sleep(wait_time)
            continue

        logger.info(f"Measurement            :: {probing_tag}")
        logger.info(f"Measurements done      :: {len(stopped_measurement_ids)}")
        logger.info(f"Measurements to insert :: {len(measurement_to_insert)}")

        # insert by batch to avoid losing some results
        for j in range(0, len(measurement_to_insert), batch_size):
            measurement_to_insert_batch = list(measurement_to_insert)[
                j : j + batch_size
            ]
            if probing_type == "ping":
                await retrieve_pings(
                    measurement_to_insert_batch,
                    measurement_table,
                    wait_time=0.1,
                    output_logs=output_logs,
                )

                await insert_geoloc_from_pings(
                    ping_table=measurement_table,
                    geoloc_table=geoloc_table,
                )

            elif probing_type == "traceroute":
                await retrieve_traceroutes(
                    measurement_to_insert,
                    measurement_table,
                    wait_time=0.1,
                    output_logs=output_logs,
                )
            else:
                raise RuntimeError(f"{probing_type} not supported")

        current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

        cached_inserted_measurements.update(measurement_to_insert)

        # stopping point for the insertion
        if len(remaining_targets) == 0:
            insert_done = True

        await asyncio.sleep(wait_time)

    logger.info("All measurements inserted")


async def insert_task(
    target_file: Path,
    in_table: str,
    out_table: str,
    experiment_uuid: str,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "insert_task.log",
    dry_run: bool = False,
    batch_size: int = 1_000,
) -> None:
    if output_logs and log_path:
        output_logs = log_path / output_logs
        setup_logger(output_logs)
    else:
        output_logs = None

    # remove targets for which we already made measurements
    cached_targets = load_cached_targets(in_table)

    # remove targets for which a measurement was started but results not inserted yet
    targets = load_csv(target_file)
    remaining_targets = set(targets).difference(set(cached_targets))

    if len(remaining_targets) == 0:
        logger.info("All measurements done, insert results process stopped")
        return
    else:
        logger.info(f"Remainning targets to geolocate:: {len(remaining_targets)}")

    if dry_run:
        logger.info("Stopped insert results task")

    await insert_results(
        targets=remaining_targets,
        probing_type="ping",
        probing_tag=experiment_uuid,
        measurement_table=in_table,
        geoloc_table=out_table,
        output_logs=output_logs,
        batch_size=batch_size,
    )

    logger.info(f"All pings and geoloc inserted, measurement finished")


# profiling, testing, debugging
if __name__ == "__main__":

    targets = load_csv(path_settings.DATASET / "demo_targets.csv")
    subnets = [get_prefix_from_ip(addr) for addr in targets]
    hostnames = load_csv(path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv")

    asyncio.run(
        insert_task(
            nb_targets=len(targets),
            ping_table="demo_ping",
            geoloc_table="demo_geoloc",
            experiment_uuid="d63c1e12-7bc4-4914-a6c0-e86e1f311338",
            output_logs=None,
        )
    )
