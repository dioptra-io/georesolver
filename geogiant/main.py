import sys
import typer
import asyncio

from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from pathlib import Path
from multiprocessing import Process

from geogiant.scores import get_scores, TargetScores
from geogiant.prober import RIPEAtlasProber
from geogiant.ripe_init import vps_init
from geogiant.ecs_mapping_init import resolve_hostnames
from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.common.files_utils import (
    create_tmp_json_file,
    load_csv,
    load_json,
)
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    load_vps,
    load_vp_subnets,
    get_subnets_mapping,
    insert_scores,
    load_target_scores,
    load_target_geoloc,
    load_cached_targets,
    insert_geoloc,
    get_subnets,
)
from geogiant.common.utils import get_parsed_vps
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import PathSettings, ClickhouseSettings, ConstantSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()
constant_settings = ConstantSettings()

BATCH_SIZE = 1_000


def parsed_score(score_per_granularity: dict, answer_granularity: str) -> list[str]:
    """parse score function of granularity and return list for insert"""
    score_data = []
    for subnet, score_per_metric in score_per_granularity.items():
        for metric, vps_subnet_score in score_per_metric.items():
            for vp_subnet, score in vps_subnet_score:
                score_data.append(
                    f"{subnet},\
                    {vp_subnet},\
                    {metric},\
                    {answer_granularity},\
                    {score}"
                )

    return score_data


async def calculate_scores(target_subnets: list[str], hostnames: list[str]) -> None:
    """calculate score and insert"""

    batch_size = 1_000  # score calculation for batch of 1000 subnets
    for i in range(0, len(target_subnets), batch_size):

        subnets = target_subnets[i : i + batch_size]
        subnet_tmp_path = create_tmp_json_file(subnets)

        score_config = {
            "targets_subnet_path": subnet_tmp_path,
            "vps_table": clickhouse_settings.VPS_FILTERED_TABLE,
            "selected_hostnames": hostnames,
            "targets_ecs_table": clickhouse_settings.TARGET_ECS_MAPPING_TABLE,
            "vps_ecs_table": clickhouse_settings.VPS_ECS_MAPPING_TABLE,
            "hostname_selection": "max_bgp_prefix",
            "score_metric": ["jaccard"],
            "answer_granularities": ["answer_subnets"],
        }

        scores: TargetScores = get_scores(score_config)
        target_score_subnet = scores.score_answer_subnets

        score_data = parsed_score(target_score_subnet, "answer_subnets")

        await insert_scores(
            csv_data=score_data,
            output_table=clickhouse_settings.TARGET_SCORE_TABLE,
        )

        subnet_tmp_path.unlink()


def get_geo_resolver_schedule(
    targets: list[str],
    subnet_scores: dict,
    vps_per_subnet: dict,
    last_mile_delay: dict,
    vps_coordinates: dict,
) -> dict:
    """get all remaining measurments for ripe ip map evaluation"""
    target_schedule = {}
    for target in tqdm(targets):
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

    return target_schedule


def load_targets_from_score(
    subnet_scores: dict[list], addr_per_subnet: dict
) -> list[str]:
    targets = []
    for target_subnet in subnet_scores:
        target_addr = addr_per_subnet[target_subnet][0]
        targets.append(target_addr)

    return targets


def get_measurement_schedule(targets: list[str], subnets: list[str]) -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )

    subnet_scores = load_target_scores(
        score_table=clickhouse_settings.TARGET_SCORE_TABLE, subnets=subnets
    )

    if not subnet_scores:
        raise RuntimeError("Should have retrieved some subnets from target_score table")

    logger.info(f"Targets in schedule:: {len(targets)}")

    target_schedule = get_geo_resolver_schedule(
        targets,
        subnet_scores,
        vps_per_subnet,
        last_mile_delay,
        vps_coordinates,
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


async def filter_ecs_subnets(subnets: list[str]) -> list[str]:
    """retrieve all subnets for which ECS resolution was done"""
    subnet_mapping = []
    subnet_mapping = get_subnets_mapping(
        dns_table=clickhouse_settings.TARGET_ECS_MAPPING_TABLE, subnets=subnets
    )

    cached_subnets = [subnet for subnet in subnet_mapping]
    filtered_subnets = set(subnets).difference(set(cached_subnets))

    return list(filtered_subnets)


async def filter_score_subnets(subnets: list[str]) -> list[str]:
    """filter and return subnets for which score calculation is yet to be done"""
    subnet_score = get_subnets(
        table_name=clickhouse_settings.TARGET_SCORE_TABLE, subnets=subnets
    )
    cached_subnets = [subnet for subnet in subnet_score]
    filtered_subnets = set(subnets).difference(set(cached_subnets))

    return list(filtered_subnets)


def filter_targets(targets: list[str], score_subnets: list[str]) -> list[str]:
    """return targets for which pings were not made"""
    filtered_targets = []

    # remove targets for which we already made measurements
    cached_targets = load_cached_targets(clickhouse_settings.TARGET_PING_TABLE)
    no_measured_target = set(targets).difference(set(cached_targets))

    logger.info(f"Number of unmeasured targets:: {len(no_measured_target)}")

    # remove targets for which we do not have scores ready
    for target in no_measured_target:
        target_subnet = get_prefix_from_ip(target)
        if target_subnet in score_subnets:
            filtered_targets.append(target)

    return filtered_targets


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


async def insert_geoloc_from_pings(targets: list[str]) -> None:
    """insert all geoloc in clickhouse"""
    target_geoloc = load_target_geoloc(
        table_name=clickhouse_settings.TARGET_PING_TABLE, targets=targets
    )

    csv_data = parse_geoloc_data(target_geoloc)

    await insert_geoloc(
        csv_data=csv_data,
        output_table=clickhouse_settings.TARGET_GEOLOC_TABLE,
    )

    return csv_data


async def ecs_mapping_task(subnets: list[str], hostname_file: list[str]) -> None:
    """run ecs mapping on batches of target subnets"""
    # Check ECS mapping exists
    filtered_subnets = await filter_ecs_subnets(subnets)
    logger.info(f"Original number of subnets:: {len(subnets)}")
    logger.info(f"Remaining subnets to geolocate:: {len(filtered_subnets)}")

    # ECS mapping
    if filtered_subnets:
        for i in range(0, len(filtered_subnets), BATCH_SIZE):
            input_subnets = filtered_subnets[i : i + BATCH_SIZE]
            logger.info(
                f"ECS mapping:: batch={(i+1) // BATCH_SIZE}/{(len(filtered_subnets) // BATCH_SIZE)}"
            )

            await resolve_hostnames(
                subnets=input_subnets,
                hostname_file=hostname_file,
                output_table=clickhouse_settings.TARGET_ECS_MAPPING_TABLE,
            )
    else:
        logger.info("Skipping ECS resolution because all subnet mapping is done")


async def score_task(
    subnets: list[str], hostnames: list[str], wait_time: int = 5, verbose: bool = False
) -> None:
    """run ecs mapping on batches of target subnets"""
    i = 0
    while True:

        # Check if ECS mapping exists for part of the subnets
        ecs_mapping_subnets = get_subnets_mapping(
            dns_table=clickhouse_settings.TARGET_ECS_MAPPING_TABLE,
            subnets=subnets,
            print_error=verbose,
        )

        if ecs_mapping_subnets:
            i = 0

            # Check if some score were already calculated
            filtered_subnets = await filter_score_subnets(subnets)
            logger.info(f"Original number of subnets:: {len(subnets)}")
            logger.info(f"Score calculation for {len(filtered_subnets)} subnets")

            # ECS mapping
            if filtered_subnets:
                for i in range(0, len(filtered_subnets), BATCH_SIZE):
                    input_subnets = filtered_subnets[i : i + BATCH_SIZE]
                    logger.info(
                        f"Score:: batch={(i+1) // BATCH_SIZE}{(len(filtered_subnets) // BATCH_SIZE)}"
                    )

                    await calculate_scores(input_subnets, hostnames)

            else:
                logger.info("Score calculation complete")
                break

        else:
            # do not polute logging
            if i == 0:
                logger.info("Waiting for ECS mapping to complete")

            i = 1
            await asyncio.sleep(wait_time)


async def geolocation_task(
    targets: list[str], subnets: list[str], verbose: bool = False, wait_time: int = 5
) -> None:
    """run ecs mapping on batches of target subnets"""
    i = 0
    while True:

        # Check if scores exists for part of the subnets
        score_subnets = get_subnets(
            table_name=clickhouse_settings.TARGET_SCORE_TABLE,
            subnets=subnets,
            print_error=verbose,
        )

        if score_subnets:
            i = 0

            logger.info(f"Found score for:: {len(score_subnets)} subnets")

            # Check if some target were already geolocated
            filtered_targets = filter_targets(targets, score_subnets)

            logger.info(f"Original number of targets:: {len(targets)}")
            logger.info(f"Geolocation for {len(filtered_targets)} targets")

            if filtered_targets:

                measurement_schedule = get_measurement_schedule(
                    filtered_targets, subnets
                )

                logger.info(
                    f"Starting geolocation for {len(measurement_schedule)} targets"
                )

                await RIPEAtlasProber(
                    probing_type="ping",
                    probing_tag="ping_targets",
                    output_table=clickhouse_settings.TARGET_PING_TABLE,
                ).main(measurement_schedule)

                logger.info("Geolocation complete")

            else:
                # do not polute logging
                if i == 0:
                    logger.info("Geolocation complete")
                    break

                i = 1

        else:
            # do not polute logging
            if i == 0:
                logger.info("Waiting for score process to complete")

            i = 1

            await asyncio.sleep(wait_time)


def main_processes(task, task_args) -> None:
    """run asynchronous task within new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(task(**task_args))


def main(
    target_file: Path,
    hostname_file: Path,
    init_vps: bool = False,
    update_meshed_pings: bool = True,
    update_meshed_traceroutes: bool = True,
    init_ecs_mapping: bool = False,
    verbose: bool = False,
) -> None:

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    if init_vps:
        logger.info(f"Starting VPs init (this step might take several days)")
        asyncio.run(vps_init(update_meshed_pings, update_meshed_traceroutes))

    if init_ecs_mapping:
        logger.info(
            f"Starting VPs ECS mapping, output table:: {clickhouse_settings.VPS_ECS_MAPPING_TABLE}"
        )

        vps_subnets = load_vp_subnets(clickhouse_settings.VPS_RAW_TABLE)
        asyncio.run(
            resolve_hostnames(
                subnets=vps_subnets,
                hostname_file=hostname_file,
                output_table=clickhouse_settings.VPS_ECS_MAPPING_TABLE,
            )
        )

    targets = load_csv(target_file)
    subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))
    hostnames = load_csv(hostname_file)

    # ecs_mapping_process = Process(
    #     target=main_processes,
    #     args=(ecs_mapping_task, {"subnets": subnets, "hostname_file": hostname_file}),
    # )
    # score_process = Process(
    #     target=main_processes,
    #     args=(score_task, {"subnets": subnets, "hostnames": hostnames}),
    # )

    geolocation_process = Process(
        target=main_processes,
        args=(
            geolocation_task,
            {"targets": targets, "subnets": subnets, "verbose": verbose},
        ),
    )

    # logger.info("Starting ECS mapping process")
    # ecs_mapping_process.start()

    # logger.info("Starting Score process")
    # score_process.start()

    logger.info("Starting Geolocation process")
    geolocation_process.start()

    # ecs_mapping_process.join()
    # score_process.join()
    geolocation_process.join()

    # TODO geolocation exists check
    # Create geolocation table/file
    # filtered_geoloc = load_geolocation()
    # await insert_geoloc_from_pings(targets)


if __name__ == "__main__":
    typer.run(main)
