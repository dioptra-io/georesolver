import asyncio

from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from pathlib import Path
from collections import defaultdict

from geogiant.prober import RIPEAtlasProber
from geogiant.clickhouse.queries import (
    get_min_rtt_per_vp,
    load_vps,
    load_target_scores,
    load_cached_targets,
    get_subnets,
)
from geogiant.common.geoloc import distance
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


def select_one_vp_per_as_city(
    raw_vp_selection: list,
    vp_coordinates: dict,
    last_mile_delay: dict,
    threshold: int = 40,
) -> list:
    """from a list of VP, select one per AS and per city"""
    filtered_vp_selection = []
    vps_per_as = defaultdict(list)
    for vp_addr, score in raw_vp_selection:
        _, _, vp_asn, _ = vp_coordinates[vp_addr]
        try:
            last_mile_delay_vp = last_mile_delay[vp_addr]
        except KeyError:
            continue

        vps_per_as[vp_asn].append((vp_addr, last_mile_delay_vp, score))

    # select one VP per AS, take maximum VP score in AS
    selected_vps_per_as = defaultdict(list)
    for asn, vps in vps_per_as.items():
        vps_per_as[asn] = sorted(vps, key=lambda x: x[1])
        for vp_i, last_mile_delay, score in vps_per_as[asn]:
            vp_i_lat, vp_i_lon, _, _ = vp_coordinates[vp_i]

            if not selected_vps_per_as[asn]:
                selected_vps_per_as[asn].append((vp_i, score))
                filtered_vp_selection.append((vp_i, score))
            else:
                already_found = False
                for vp_j, score in selected_vps_per_as[asn]:

                    vp_j_lat, vp_j_lon, _, _ = vp_coordinates[vp_j]

                    d = distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

                    if d < threshold:
                        already_found = True
                        break

                if not already_found:
                    selected_vps_per_as[asn].append((vp_i, score))
                    filtered_vp_selection.append((vp_i, score))

    return filtered_vp_selection


def get_ecs_vps(
    target_subnet: str,
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
    # target_score = sorted(target_score, key=lambda x: x[1], reverse=True)
    for subnet, score in target_score:
        # for fairness, do not take vps that are in the same subnet as the target
        if subnet == target_subnet:
            continue

        vps_in_subnet = vps_per_subnet[subnet]

        vps_delay_subnet = []
        for vp in vps_in_subnet:
            try:
                vps_delay_subnet.append((vp, last_mile_delay_vp[vp]))
            except KeyError:
                continue

        # for each subnet, elect the VP with the lowest last mile delay
        if vps_delay_subnet:
            elected_subnet_vp_addr, _ = min(vps_delay_subnet, key=lambda x: x[-1])
            ecs_vps.append((elected_subnet_vp_addr, score))

        # take only a number of subnets up to probing budget
        if len(ecs_vps) >= probing_budget:
            break

    return ecs_vps


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
    vps = load_vps(clickhouse_settings.VPS_FILTERED_FINAL_TABLE)
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


async def ping_task(
    target_file: Path,
    in_table: str,
    out_table: str,
    agent_uuid: str,
    wait_time: int = 30,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "ping_task.log",
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

    geolocated_targets = []
    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=agent_uuid,
        output_table=out_table,
        output_logs=output_logs,
    )
    while True:

        # Retrieve target with no geolocation for which we have a score
        filtered_targets, no_measured_target = filter_targets(
            targets=targets,
            geolocated_targets=geolocated_targets,
            score_table=in_table,
            ping_table=out_table,
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
                score_table=in_table,
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
        ping_task(
            targets=targets,
            score_table="demo_score",
            ping_table="demo_ping",
            agent_uuid="d63c1e12-7bc4-4914-a6c0-e86e1f311338",
            output_logs=None,
        )
    )
