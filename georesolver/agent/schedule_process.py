"""retrieve results from score process and create ping schedule"""

import asyncio

from tqdm import tqdm
from loguru import logger
from pathlib import Path
from collections import defaultdict

from georesolver.clickhouse.queries import (
    get_min_rtt_per_vp,
    load_vps,
    load_target_scores,
    get_subnets,
    insert_schedule,
)
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_csv
from georesolver.common.utils import get_vps_per_subnet
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    ConstantSettings,
    setup_logger,
)

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()
constant_settings = ConstantSettings()

BATCH_SIZE = 1_000


def select_one_vp_per_as_city(
    raw_vp_selection: list,
    last_mile_delay: dict,
    threshold: int = 40,
) -> list:
    """from a list of VP, select one per AS and per city"""
    filtered_vp_selection = []
    vps_per_as = defaultdict(list)
    for vp, score in raw_vp_selection:
        try:
            last_mile_delay_vp = last_mile_delay[vp["addr"]]
        except KeyError:
            continue

        vps_per_as[vp["asn"]].append((vp, last_mile_delay_vp, score))

    # select one VP per AS, take maximum VP score in AS
    selected_vps_per_as = defaultdict(list)
    for asn, vps in vps_per_as.items():
        vps_per_as[asn] = sorted(vps, key=lambda x: x[-1], reverse=True)
        for vp_i, last_mile_delay, score in vps_per_as[asn]:
            if not selected_vps_per_as[asn]:
                selected_vps_per_as[asn].append((vp_i, score))
                filtered_vp_selection.append((vp_i, score))
            else:
                already_found = False
                for vp_j, score in selected_vps_per_as[asn]:

                    d = distance(vp_i["lat"], vp_j["lat"], vp_i["lon"], vp_j["lon"])

                    if d < threshold:
                        already_found = True
                        break

                if not already_found:
                    selected_vps_per_as[asn].append((vp_i, score))
                    filtered_vp_selection.append((vp_i, score))

    return filtered_vp_selection


def get_ecs_vps(
    target_score: dict,
    vps_per_subnet: dict,
    last_mile_delay_vp: dict,
    probing_budget: int = 50,
) -> list:
    """
    get the target score and extract best VPs function of the probing budget
    """
    # retrieve all vps belonging to subnets with highest mapping scores
    ecs_vps = []
    target_score = sorted(target_score, key=lambda x: x[1], reverse=True)
    for subnet, score in target_score:
        vps_in_subnet = vps_per_subnet[subnet]

        vps_delay_subnet = []
        for vp in vps_in_subnet:
            try:
                vps_delay_subnet.append((vp, last_mile_delay_vp[vp["addr"]]))
            except KeyError:
                continue

        # for each subnet, elect the VP with the lowest last mile delay
        if vps_delay_subnet:
            elected_subnet_vp, _ = min(vps_delay_subnet, key=lambda x: x[-1])
            ecs_vps.append((elected_subnet_vp, score))

        # take only a number of subnets up to probing budget
        if len(ecs_vps) >= probing_budget:
            break

    return ecs_vps


def get_georesolver_schedule(
    subnet_scores: dict,
    vps_per_subnet: dict,
    last_mile_delay: dict,
    output_logs: Path = None,
) -> dict:
    """get all remaining measurments for ripe ip map evaluation"""
    schedule = {}

    if output_logs:
        output_file = output_logs.open("a")
    else:
        output_file = None

    for subnet, subnet_scores in tqdm(subnet_scores.items(), file=output_file):
        ecs_vps = get_ecs_vps(subnet_scores, vps_per_subnet, last_mile_delay, 5_00)
        ecs_vps = select_one_vp_per_as_city(ecs_vps, last_mile_delay)
        schedule[subnet] = ecs_vps[: constant_settings.PROBING_BUDGET]

    if output_file:
        output_file.close()

    return schedule


def load_targets_from_score(
    subnet_scores: dict[list], addr_per_subnet: dict
) -> list[str]:
    targets = []
    for target_subnet in subnet_scores:
        target_addr = addr_per_subnet[target_subnet][0]
        targets.append(target_addr)

    return targets


def filter_subnets(
    subnets: list[str],
    score_table: str,
    schedule_table: str,
    verbose: bool = False,
) -> list[str]:
    """return targets for which pings were not made"""

    # Check if scores exists for part of the subnets
    cached_score_subnets = get_subnets(table_name=score_table, print_error=verbose)
    if not cached_score_subnets:
        return [], subnets

    # retrieve all subnets for which we have a schedule and filter
    cached_schedule_subnets = get_subnets(schedule_table, print_error=verbose)
    no_schedule_subnet = set(subnets).difference(set(cached_schedule_subnets))
    filtered_subnets = set(no_schedule_subnet).intersection(set(cached_score_subnets))

    logger.info(f"Found {len(cached_score_subnets)} subnets with score")

    return list(filtered_subnets), list(no_schedule_subnet)


async def insert_measurement_schedule(
    subnets: list[str],
    score_table: str,
    schedule_table: str,
    output_logs: Path = None,
) -> dict[list]:
    """calculate distance error and latency for each score"""
    vps = load_vps(clickhouse_settings.VPS_FILTERED_FINAL_TABLE)
    vps_per_subnet = get_vps_per_subnet(vps)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )

    subnet_scores = {}
    logger.info(f"Retriving scores for {len(subnets)}")
    batch_score_size = 5_000
    for i in range(0, len(subnets), batch_score_size):
        logger.info(f"Batch:: {i} -> {i + batch_score_size}")
        batch_subnets = subnets[i : i + batch_score_size]
        subnet_scores.update(
            load_target_scores(score_table=score_table, subnets=batch_subnets)
        )

    schedule = get_georesolver_schedule(
        subnet_scores,
        vps_per_subnet,
        last_mile_delay,
        output_logs=output_logs,
    )

    csv_data = []
    for subnet, vps_schedule in schedule.items():
        for vp, score in vps_schedule:
            csv_data.append(f"{subnet},{vp['id']},{vp['addr']},{vp['subnet']},{score}")

    await insert_schedule(csv_data, schedule_table)


async def schedule_task(
    target_file: Path,
    in_table: str,
    out_table: str,
    wait_time: int = 30,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "schedule.log",
    dry_run: bool = False,
) -> None:
    """run GeoResolver on batches of target subnets"""
    if output_logs and log_path:
        output_logs = log_path / output_logs
        setup_logger(output_logs)
    else:
        output_logs = None

    logger.info("Targets loading")
    targets = load_csv(target_file, exit_on_failure=True)

    while True:

        # Retrieve target with no geolocation for which we have a score
        subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))
        filtered_subnets, no_schedule_subnets = filter_subnets(
            subnets=subnets,
            score_table=in_table,
            schedule_table=out_table,
        )

        if not no_schedule_subnets:
            logger.info("No remaning target to geolocate, process finished")
            break

        if filtered_subnets:

            logger.info(f"Running schedule for {len(filtered_subnets)} subnets")

            if dry_run:
                logger.info("Stopped Geolocation process")
                break

            max_batch_size = 50_000
            for i in range(0, len(filtered_subnets), max_batch_size):
                logger.info(f"Batch:: {i} -> {i+max_batch_size}")
                batch_subnets = filtered_subnets[i : i + max_batch_size]

                # get measurement schedule for all subnets with score
                await insert_measurement_schedule(
                    subnets=batch_subnets,
                    score_table=in_table,
                    schedule_table=out_table,
                    output_logs=output_logs,
                )

        else:
            logger.info(
                f"Waiting for score process to complete, remaining targets :: {len(no_schedule_subnets)}"
            )
            await asyncio.sleep(wait_time)

            if dry_run:
                logger.info("Stopped Geolocation process")
                break


if __name__ == "__main__":
    asyncio.run(
        schedule_task(
            target_file=path_settings.DATASET
            / "itdk/itdk_responsive_router_interface_parsed.csv",
            in_table="itdk_score",
            out_table="itdk_schedule",
        )
    )
