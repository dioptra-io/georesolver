import sys
import typer
import asyncio

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
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.common.utils import get_parsed_vps
from geogiant.common.files_utils import create_tmp_json_file, load_csv, load_json
from geogiant.common.queries import get_min_rtt_per_vp, load_vps
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings, ClickhouseSettings, ConstantSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()
constant_settings = ConstantSettings()


async def calculate_scores(hostnames: list[str]) -> None:

    selected_hostnames_per_cdn, selected_hostnames = load_hostnames()
    score_subnets = load_subnets()

    batch_size = 1_000  # score calculation for batch of 1000 subnets
    for i in range(0, len(score_subnets), batch_size):
        output_path = (
            path_settings.RESULTS_PATH
            / f"internet_scale_evaluation/score__internet_scale_{uuid4()}.pickle"
        )
        subnets = score_subnets[i : i + batch_size]
        subnet_tmp_path = create_tmp_json_file(subnets)

        score_config = {
            "targets_subnet_path": subnet_tmp_path,
            "vps_subnet_path": VPS_SUBNET_PATH,
            "hostname_per_cdn": selected_hostnames_per_cdn,
            "selected_hostnames": selected_hostnames,
            "targets_ecs_table": ECS_TABLE,
            "vps_ecs_table": VPS_ECS_TABLE,
            "hostname_selection": "max_bgp_prefix",
            "score_metric": ["jaccard"],
            "answer_granularities": ["answer_subnets"],
            "output_path": output_path,
        }

        get_scores(score_config)

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
            logger.error(f"{target_subnet} does not have score")
        for _, target_score in target_scores.items():

            # get vps, function of their subnet ecs score
            ecs_vps = get_ecs_vps(
                target_subnet, target_score, vps_per_subnet, last_mile_delay, 5_00
            )

            # remove vps that have a high last mile delay
            ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

            ecs_vps = select_one_vp_per_as_city(
                ecs_vps, vps_coordinates, last_mile_delay
            )[: constant_settings.PROBING_BUDGET]

            target_schedule[target] = [vp_addr for vp_addr, _ in ecs_vps]

    return target_schedule


def get_measurement_schedule() -> dict[list]:
    """calculate distance error and latency for each score"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )

    internet_scale_dataset = load_json(path_settings.USER_HITLIST_FILE)
    # TODO: calculate score + table
    subnet_scores = load_subnet_scores()
    targets = load_targets_from_score(subnet_scores, internet_scale_dataset)

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


async def print_name(name: str) -> None:
    logger.info(f"Hello {name}")


async def geolocate(
    target_file: Path, hostname_file: Path = None, verbose: bool = False
) -> list[tuple]:
    "main function of GeoResolver, get IP addresses and perform geolocation"
    ip_addrs = load_csv(target_file)
    subnets = [get_prefix_from_ip(ip) for ip in ip_addrs]

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

    # 1. checking phase
    # TODO

    # 2. perform ECS queries
    await ecs_mapping(
        subnets=subnets,
        hostnames=hostnames,
        ecs_table=clickhouse_settings.ECS_TARGET_TABLE,
    )

    # geoloc


def main(target_file: Path, hostname_file: Path = None):
    asyncio.run(geolocate(target_file, hostname_file))


if __name__ == "__main__":
    typer.run(main)
