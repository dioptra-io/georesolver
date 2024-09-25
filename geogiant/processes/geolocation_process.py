import asyncio

from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from pathlib import Path

from geogiant.prober import RIPEAtlasProber
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.common.files_utils import (
    load_json,
)
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    load_vps,
    load_target_scores,
    load_target_geoloc,
    load_cached_targets,
    insert_geoloc,
    get_subnets,
)
from geogiant.common.utils import get_parsed_vps
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import (
    PathSettings,
    ClickhouseSettings,
    ConstantSettings,
    setup_logger,
)

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()
constant_settings = ConstantSettings()

BATCH_SIZE = 1_000


def get_geo_resolver_schedule(
    targets: list[str],
    subnet_scores: dict,
    vps_per_subnet: dict,
    last_mile_delay: dict,
    vps_coordinates: dict,
    output_logs: Path = None,
) -> dict:
    """get all remaining measurments for ripe ip map evaluation"""
    target_schedule = {}

    if output_logs:
        output_file = output_logs.open("a")
    else:
        output_file = None

    for target in tqdm(targets, file=output_file):
        target_subnet = get_prefix_from_ip(target)
        target_scores = subnet_scores[target_subnet]

        if not target_scores:
            raise RuntimeError(f"{target_subnet} does not have score")

        # get vps, function of their subnet ecs score
        ecs_vps = get_ecs_vps(
            target_subnet, target_scores, vps_per_subnet, last_mile_delay, 5_00
        )

        # remove vps that have a high last mile delay
        ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

        ecs_vps = select_one_vp_per_as_city(ecs_vps, vps_coordinates, last_mile_delay)[
            : constant_settings.PROBING_BUDGET
        ]

        target_schedule[target] = [vp_addr for vp_addr, _ in ecs_vps]

    if output_file:
        output_file.close()

    return target_schedule


def load_targets_from_score(
    subnet_scores: dict[list], addr_per_subnet: dict
) -> list[str]:
    targets = []
    for target_subnet in subnet_scores:
        target_addr = addr_per_subnet[target_subnet][0]
        targets.append(target_addr)

    return targets


def get_measurement_schedule(
    targets: list[str],
    subnets: list[str],
    score_table: str,
    output_logs: Path = None,
) -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )

    subnet_scores = load_target_scores(score_table=score_table, subnets=subnets)

    if not subnet_scores:
        raise RuntimeError(
            f"Should have retrieved some subnets from {score_table} table"
        )

    logger.info(f"Targets in schedule:: {len(targets)}")

    target_schedule = get_geo_resolver_schedule(
        targets,
        subnet_scores,
        vps_per_subnet,
        last_mile_delay,
        vps_coordinates,
        output_logs=output_logs,
    )

    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    measurement_schedule = []
    for target, vps in target_schedule.items():
        measurement_schedule.append(
            (target, [vp_id_per_addr[vp_addr] for vp_addr in vps])
        )

    return measurement_schedule


def filter_targets(
    targets: list[str],
    score_subnets: list[str],
    ping_table: str,
) -> list[str]:
    """return targets for which pings were not made"""
    filtered_targets = []

    # remove targets for which we already made measurements
    cached_targets = load_cached_targets(ping_table)
    no_measured_target = set(targets).difference(set(cached_targets))

    logger.info(f"Number of unmeasured targets:: {len(no_measured_target)}")

    # remove targets for which we do not have scores ready
    for target in no_measured_target:
        target_subnet = get_prefix_from_ip(target)
        if target_subnet in score_subnets:
            filtered_targets.append(target)

    return filtered_targets, no_measured_target


def parse_geoloc_data(target_geoloc: dict) -> list[str]:
    """parse ping data to geoloc csv"""
    csv_data = []
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    _, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    for target_addr, shortest_ping_data in target_geoloc.items():
        target_subnet = get_prefix_from_ip(target_addr)
        target_asn, target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)
        if not target_bgp_prefix or not target_asn:
            target_bgp_prefix = "Unknown"
            target_asn = -1

        vp_addr = shortest_ping_data[0]
        vp_subnet = get_prefix_from_ip(vp_addr)
        vp_asn, vp_bgp_prefix = route_view_bgp_prefix(vp_addr, asndb)

        if not vp_bgp_prefix or not vp_asn:
            vp_bgp_prefix = "Unknown"
            vp_asn = -1

        lat, lon, country_code, _ = vps_coordinates[vp_addr]
        msm_id = shortest_ping_data[1]
        min_rtt = shortest_ping_data[2]

        csv_data.append(
            f"{target_addr},\
            {target_subnet},\
            {target_bgp_prefix},\
            {target_asn},\
            {lat},\
            {lon},\
            {country_code},\
            {vp_addr},\
            {vp_subnet},\
            {vp_bgp_prefix},\
            {vp_asn},\
            {min_rtt},\
            {msm_id}"
        )

    return csv_data


async def insert_geoloc_from_pings(
    targets: list[str], ping_table: str, geoloc_table: str
) -> None:
    """insert all geoloc in clickhouse"""
    target_geoloc = load_target_geoloc(table_name=ping_table, targets=targets)

    csv_data = parse_geoloc_data(target_geoloc)

    await insert_geoloc(
        csv_data=csv_data,
        output_table=geoloc_table,
    )

    return csv_data


async def geolocation_task(
    targets: list[str],
    subnets: list[str],
    score_table: str,
    ping_table: str,
    verbose: bool = False,
    wait_time: int = 30,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "geolocation_task.log",
) -> None:
    """run ecs mapping on batches of target subnets"""
    setup_logger(log_path / output_logs)

    while True:

        # Check if scores exists for part of the subnets
        score_subnets = get_subnets(
            table_name=score_table,
            subnets=subnets,
            print_error=verbose,
        )

        if score_subnets:

            logger.info(f"Found score for:: {len(score_subnets)} subnets")

            # Check if some target were already geolocated
            filtered_targets, no_measured_target = filter_targets(
                targets, score_subnets, ping_table
            )

            logger.info(f"Original number of targets:: {len(targets)}")
            logger.info(f"Remaining target to geolocate:: {len(no_measured_target)}")

            if not no_measured_target:
                logger.info("No remaning target to geolocate, process finished")
                break

            if filtered_targets:

                logger.info(f"Running geolocation for {len(filtered_targets)} targets")

                measurement_schedule = get_measurement_schedule(
                    targets=filtered_targets,
                    subnets=subnets,
                    score_table=score_table,
                    output_logs=log_path / output_logs,
                )

                logger.info(
                    f"Starting geolocation for {len(measurement_schedule)} targets"
                )

                await RIPEAtlasProber(
                    probing_type="ping",
                    probing_tag="ping_targets",
                    output_table=ping_table,
                    output_logs=log_path / output_logs,
                ).main(measurement_schedule)

                logger.info("Geolocation complete")

            else:
                logger.info("Waiting for score process to complete")
                await asyncio.sleep(wait_time)

        else:
            logger.info("Waiting for score process to complete")
            await asyncio.sleep(wait_time)
