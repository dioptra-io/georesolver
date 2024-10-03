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
from geogiant.common.queries import (
    get_min_rtt_per_vp,
    load_vps,
    load_target_scores,
    load_cached_targets,
    get_subnets,
)
from geogiant.common.files_utils import load_csv
from geogiant.common.utils import get_parsed_vps
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
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
    subnet_scores = {}
    for i in range(0, len(subnets), 1000):
        batch_subnets = subnets[i : i + 1_000]
        subnet_scores.update(
            load_target_scores(score_table=score_table, subnets=batch_subnets)
        )

    if not subnet_scores:
        raise RuntimeError(
            f"Should have retrieved some subnets from {score_table} table"
        )

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
    geolocated_targets: list[str],
    score_table: str,
    ping_table: str,
    verbose: bool = False,
) -> list[str]:
    """return targets for which pings were not made"""
    filtered_targets = []

    # Check if scores exists for part of the subnets
    cached_score_subnets = get_subnets(
        table_name=score_table,
        print_error=verbose,
    )

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


async def geolocation_task(
    targets: list[str],
    score_table: str,
    ping_table: str,
    measurement_uuid: str,
    wait_time: int = 30,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "geolocation_task.log",
    dry_run: bool = False,
) -> None:
    """run GeoResolver on batches of target subnets"""
    if output_logs and log_path:
        output_logs = log_path / output_logs
        setup_logger(output_logs)
    else:
        output_logs = None

    geolocated_targets = []
    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=measurement_uuid,
        output_table=ping_table,
        output_logs=output_logs,
    )
    while True:

        # Retrieve target with no geolocation for which we have a score
        filtered_targets, no_measured_target = filter_targets(
            targets=targets,
            geolocated_targets=geolocated_targets,
            score_table=score_table,
            ping_table=ping_table,
        )

        logger.info(f"Original number of targets:: {len(targets)}")
        logger.info(f"Remaining target to geolocate:: {len(no_measured_target)}")

        if not no_measured_target:
            logger.info("No remaning target to geolocate, process finished")
            break

        if filtered_targets:

            logger.info(f"Running geolocation for {len(filtered_targets)} targets")

            if dry_run:
                logger.info("Stopped Geolocation process")
                break

            # get measurement schedule for all subnets with score
            measurement_schedule = get_measurement_schedule(
                targets=filtered_targets,
                subnets=[get_prefix_from_ip(target) for target in filtered_targets],
                score_table=score_table,
                output_logs=output_logs,
            )

            logger.info(
                f"Starting geolocation round for {len(measurement_schedule)} targets"
            )

            await prober.main(measurement_schedule)

            geolocated_targets.extend(filtered_targets)

            logger.info("Geolocation round complete")

        else:
            logger.info("Waiting for score process to complete")
            await asyncio.sleep(wait_time)

            if dry_run:
                logger.info("Stopped Geolocation process")
                break


# profiling, testing, debugging
if __name__ == "__main__":

    targets = load_csv(path_settings.DATASET / "demo_targets.csv")
    subnets = [get_prefix_from_ip(addr) for addr in targets]
    hostnames = load_csv(path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv")

    asyncio.run(
        geolocation_task(
            targets=targets,
            score_table="demo_score",
            ping_table="demo_ping",
            measurement_uuid="d63c1e12-7bc4-4914-a6c0-e86e1f311338",
            output_logs=None,
        )
    )
