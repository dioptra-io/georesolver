import sys
import typer
import time
import asyncio

from uuid import uuid4
from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from pathlib import Path

from geogiant.evaluation.ecs_geoloc_eval import (
    get_ecs_vps,
    filter_vps_last_mile_delay,
    select_one_vp_per_as_city,
)
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.ecs_vp_selection.scores import get_scores, TargetScores
from geogiant.prober import RIPEAtlasProber
from geogiant.common.utils import get_parsed_vps
from geogiant.common.files_utils import (
    create_tmp_json_file,
    load_csv,
    load_json,
    load_pickle,
)
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    load_vps,
    get_subnets_mapping,
    insert_scores,
    load_target_scores,
    retrieve_pings,
    load_target_geoloc,
    load_cached_targets,
    insert_geoloc,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import PathSettings, ClickhouseSettings, ConstantSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()
constant_settings = ConstantSettings()


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
        output_path = (
            path_settings.RESULTS_PATH
            / f"internet_scale_evaluation/score__geoResolver_5468f310-c23e-45f2-b19b-c86b796a043e.pickle"
        )
        subnets = target_subnets[i : i + batch_size]
        subnet_tmp_path = create_tmp_json_file(subnets)

        score_config = {
            "targets_subnet_path": subnet_tmp_path,
            "vps_table": clickhouse_settings.VPS_FILTERED_TABLE,
            "selected_hostnames": hostnames,
            "targets_ecs_table": clickhouse_settings.ECS_TARGET_TABLE,
            "vps_ecs_table": clickhouse_settings.ECS_VPS_TABLE,
            "hostname_selection": "max_bgp_prefix",
            "score_metric": ["jaccard"],
            "answer_granularities": ["answer_subnets"],
            "output_path": output_path,
        }

        scores: TargetScores = get_scores(score_config)
        # scores: TargetScores = load_pickle(output_path)
        target_score_subnet = scores.score_answer_subnets

        score_data = parsed_score(target_score_subnet, "answer_subnets")

        await insert_scores(
            csv_data=score_data,
            output_table=clickhouse_settings.SCORE_TARGET_TABLE,
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
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )

    subnet_scores = load_target_scores(
        score_table=clickhouse_settings.SCORE_TARGET_TABLE, subnets=subnets
    )

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


async def ecs_mapping(subnets: list[str], hostnames: list[str], ecs_table: str) -> None:
    batch_size = 10_000
    for i in range(0, len(subnets), batch_size):
        logger.info(f"Batch:: {i+1}/{len(subnets) // batch_size}")
        batch_subnets = subnets[i : i + batch_size]

        subnet_tmp_file_path = create_tmp_json_file(batch_subnets)

        await resolve_vps_subnet(
            selected_hostnames=hostnames,
            input_file=subnet_tmp_file_path,
            output_table=ecs_table,
            chunk_size=500,
        )

        subnet_tmp_file_path.unlink()


async def filter_ecs_subnets(subnets: list[str]) -> list[str]:
    """retrieve all subnets for which ECS resolution was done"""
    subnet_mapping = []
    subnet_mapping = get_subnets_mapping(
        dns_table=clickhouse_settings.ECS_TARGET_TABLE, subnets=subnets
    )

    cached_subnets = [subnet for subnet in subnet_mapping]
    filtered_subnets = set(subnets).difference(set(cached_subnets))

    return list(filtered_subnets)


async def filter_score_subnets(subnets: list[str]) -> list[str]:
    """filter and return subnets for which score calculation is yet to be done"""
    subnet_score = load_target_scores(
        score_table=clickhouse_settings.SCORE_TARGET_TABLE, subnets=subnets
    )
    cached_subnets = [subnet for subnet in subnet_score]
    filtered_subnets = set(subnets).difference(set(cached_subnets))

    return list(filtered_subnets)


def filter_targets(targets: list[str]) -> list[str]:
    """return targets for which pings were not made"""
    cached_targets = load_cached_targets(clickhouse_settings.PING_TARGET_TABLE)
    filtered_targets = set(targets).difference(set(cached_targets))

    return list(filtered_targets)


async def ping_targets(measurement_schedule: list, wait_time: int = 60 * 20) -> None:
    """perform ping geolocation"""
    if measurement_schedule:

        batch_size = 1_000
        for i in range(0, len(measurement_schedule), batch_size):

            batch_schedule = measurement_schedule[i : i + batch_size]

            logger.debug(
                f"Pings schedule (batch {(i + 1) // batch_size}/{len(measurement_schedule) // batch_size})"
            )

            output_path = await RIPEAtlasProber(
                probing_type="ping", probing_tag="ping_targets"
            ).main(batch_schedule)

            measurement_config = load_json(output_path)
            measurement_ids = measurement_config["ids"]

            # TODO: wait for batch of measurement to be done instead of wait time
            time.sleep(60 * 5)

            await insert_measurements(
                measurement_ids=measurement_ids,
                ping_table=clickhouse_settings.PING_TARGET_TABLE,
            )

            output_path.unlink()


def parse_geoloc_data(target_geoloc: dict) -> list[str]:
    """parse ping data to geoloc csv"""
    csv_data = []
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
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


async def insert_measurements(measurement_ids: list[int], ping_table: str) -> None:
    logger.info(f"{len(measurement_ids)} measurements to insert")

    logger.info(f"Retreiving results for {len(measurement_ids)} measurements")

    batch_size = 100
    for i in range(0, len(measurement_ids), batch_size):
        ids = measurement_ids[i : i + batch_size]
        await retrieve_pings(ids, ping_table)


async def insert_geoloc_from_pings(targets: list[str]) -> None:
    """insert all geoloc in clickhouse"""
    target_geoloc = load_target_geoloc(
        table_name=clickhouse_settings.PING_TARGET_TABLE, targets=targets
    )

    csv_data = parse_geoloc_data(target_geoloc)

    await insert_geoloc(
        csv_data=csv_data,
        output_table=clickhouse_settings.GEOLOC_TARGET_TABLE,
    )

    return csv_data


async def geolocate(
    target_file: Path,
    hostname_file: Path = None,
    verbose: bool = False,
    output_file: Path = None,
) -> list[tuple]:
    "main function of GeoResolver, get IP addresses and perform geolocation"
    targets = load_csv(target_file)
    subnets = [get_prefix_from_ip(ip) for ip in targets]

    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    if not hostname_file:
        hostnames = load_csv(path_settings.DEFAULT_HOSTNAME_FILE)
    else:
        hostnames = load_csv(hostname_file)

    # 1. checking phase: 1) geoloc, 2) ECS mapping, 3) score check
    # TODO geolocation exists check

    # Check ECS mapping exists
    filtered_subnets = await filter_ecs_subnets(subnets)
    logger.info(f"Original number of subnets:: {len(subnets)}")
    logger.info(f"Remaining subnets to geolocate:: {len(filtered_subnets)}")

    # ECS mapping
    if filtered_subnets:
        await ecs_mapping(
            subnets=filtered_subnets,
            hostnames=hostnames,
            ecs_table=clickhouse_settings.ECS_TARGET_TABLE,
        )
    else:
        logger.info("Skipping ECS resolution because all subnet mapping is done")

    # Check score
    filtered_subnet_score = await filter_score_subnets(subnets)
    logger.info(f"Original number of subnets:: {len(subnets)}")
    logger.info(f"Score calculation for {len(filtered_subnet_score)} subnets")

    # Calculate scores
    if filtered_subnet_score:
        await calculate_scores(filtered_subnet_score, hostnames)
    else:
        logger.info(f"Skipping score calculation bacause all score subnet is available")

    # Check pings
    filtered_targets = filter_targets(targets)

    logger.info(f"Original number of targets:: {len(targets)}")
    logger.info(f"Pings for {len(filtered_targets)} targets")

    if filtered_targets:
        # Pings
        measurement_schedule = get_measurement_schedule(filtered_targets, subnets)

        logger.info(f"Starting geolocation for {len(measurement_schedule)} targets")

        await ping_targets(measurement_schedule, clickhouse_settings.PING_TARGET_TABLE)

        logger.info("Geolocation done, retrieving measurements")
    else:
        logger.info("Skipping pings because they already exists")

    # Create geolocation table/file
    filtered_geoloc = load_geolocation()
    await insert_geoloc_from_pings(targets)


def main(target_file: Path, hostname_file: Path = None):
    asyncio.run(geolocate(target_file, hostname_file))


if __name__ == "__main__":
    typer.run(main)
