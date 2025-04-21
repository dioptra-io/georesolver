"""study and evaluate CDNs redirection strategies"""

import asyncio

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from random import choice
from collections import defaultdict
from datetime import datetime, timedelta

from georesolver.clickhouse.queries import (
    load_vps,
    get_mapping_answers,
    get_measurement_ids,
    get_pings_per_target_extended,
)
from georesolver.zdns.zmap import zmap
from georesolver.agent.insert_process import retrieve_pings
from georesolver.prober import RIPEAtlasProber, RIPEAtlasAPI
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.files_utils import load_csv, dump_csv, load_json
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

MEASURMENT_TAG = "meshed-cdns-pings-test"
OUTPUT_TABLE = "meshed_cdns_pings_test"

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
ripe_atlas_settings = RIPEAtlasSettings()


def get_responsive_answers(answers: list[str], output_path: Path) -> list[str]:
    """get responsive IP addresses from a list of pings"""

    if not output_path.exists():
        subnets = list(set([get_prefix_from_ip(answer) for answer in answers]))
        responsive_answers = zmap(subnets)

        dump_csv(responsive_answers, output_path)

        return responsive_answers

    responsive_answers = load_csv(output_path)

    return responsive_answers


def get_measurement_schedule() -> list[tuple[str, list]]:
    """perform meshed pings towards all CDNs replicas present in VPs mapping"""
    responsive_answers_path = path_settings.DATASET / "responsive_answers_cdns.csv"

    # 1. load all VPs mapping answers
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    answers = get_mapping_answers(ch_settings.VPS_ECS_MAPPING_TABLE)
    all_subnets = set([get_prefix_from_ip(answer) for answer in answers])

    logger.info(f"All answers :: {len(answers)}")
    logger.info(f"All subnets :: {len(all_subnets)}")

    # 2. filter based on responsive answers
    responsive_answers = get_responsive_answers(answers, responsive_answers_path)
    answers = set(answers).intersection(set(responsive_answers))

    logger.info(f"Responsive answers :: {len(answers)}")

    # 3. get answers per subnets
    answer_per_subnets = defaultdict(set)
    for answer in answers:
        subnet = get_prefix_from_ip(answer)
        answer_per_subnets[subnet].add(answer)

    logger.info(f"Total number of subnets to probe {len(all_subnets)}")

    # 3. select one IP address per /24
    measurement_schedule = []
    all_targets = set()
    for subnet, answers in answer_per_subnets.items():
        # select one IP randomly
        target = choice(list(answers))

        # prepare schedule
        batch_size = ripe_atlas_settings.MAX_VP
        for i in range(0, len(vps), batch_size):
            batch_vps = [vp["id"] for vp in vps[i : i + batch_size]]
            measurement_schedule.append((target, batch_vps))

        all_targets.add(target)

    logger.info(f"Total Nb measurement to run:: {len(measurement_schedule)}")

    return measurement_schedule


async def insert_measurements(
    measurement_schedule: list[tuple],
    probing_tags: list[str],
    output_table: str,
    wait_time: int = 60,
) -> None:
    """insert measurement once they are tagged as Finished on RIPE Atlas"""
    current_time = datetime.timestamp(datetime.now() - timedelta(days=2))
    cached_measurement_ids = set()
    while True:
        # load measurement finished from RIPE Atlas
        stopped_measurement_ids = await RIPEAtlasAPI().get_stopped_measurement_ids(
            start_time=current_time, tags=probing_tags
        )

        # load already inserted measurement ids
        inserted_ids = get_measurement_ids(output_table)

        # stop measurement once all measurement are inserted
        all_measurement_ids = set(inserted_ids).union(cached_measurement_ids)
        if len(all_measurement_ids) >= len(measurement_schedule):
            logger.info(
                f"All measurement inserted:: {len(inserted_ids)=}; {len(measurement_schedule)=}"
            )
            break

        measurement_to_insert = set(stopped_measurement_ids).difference(
            set(inserted_ids)
        )

        # check cached measurements,
        # some measurement are not insersed because no results
        measurement_to_insert = set(measurement_to_insert).difference(
            cached_measurement_ids
        )

        logger.info(f"{len(stopped_measurement_ids)=}")
        logger.info(f"{len(inserted_ids)=}")
        logger.info(f"{len(measurement_to_insert)=}")

        if not measurement_to_insert:
            await asyncio.sleep(wait_time)
            continue

        # insert measurement
        await retrieve_pings(measurement_to_insert, output_table)

        cached_measurement_ids.update(measurement_to_insert)
        current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

        await asyncio.sleep(wait_time)


async def meshed_ping_cdns(prev_schedule: Path) -> None:
    """
    perform pings towards one IP address
    for each /24 prefix present in VPs ECS mapping redirection
    """

    if not prev_schedule:
        measurement_schedule = get_measurement_schedule()
    else:
        measurement_schedule = []

        # filter based on existing schedule/meaasurement
        cached_measurement_schedule = load_json(prev_schedule)
        # load existing measurement
        cached_measurements = get_pings_per_target_extended(OUTPUT_TABLE)

        logger.info(f"{len(cached_measurements)=}")

        for target, vp_ids in tqdm(cached_measurement_schedule):
            # retrieve pings for target, if exists
            try:
                cached_ping = cached_measurements[target]
            except KeyError:
                continue

            # only keep vp ids not in cached measurements
            cached_vp_ids = [vp_id for _, vp_id, _ in cached_ping]
            remaning_vp_ids = list(set(vp_ids).difference(cached_vp_ids))

            measurement_schedule.append((target, remaning_vp_ids))

        logger.info(f"{len(measurement_schedule)=}")

    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=MEASURMENT_TAG,
        output_table=OUTPUT_TABLE,
        protocol="ICMP",
    )

    await asyncio.gather(
        prober.main(measurement_schedule),
        insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", MEASURMENT_TAG],
            output_table=OUTPUT_TABLE,
        ),
    )

    logger.info("Meshed CDNs pings measurement done")


def latency_eval() -> None:
    """evaluate if the redirected PoP was the most optimal based on ping measurements"""
    pass


def geo_eval() -> None:
    """check if the redirected PoP was the closest one or not"""
    pass


def main() -> None:
    """
    entry point:
        - run meshed measurements per CDNs (all VPs towards one IP addr per /24 answer)
        - evaluate redirection latency
        - evaluate max distance based on exact position + geolocation area of presence
    """
    do_measurement: bool = True
    do_latency_eval: bool = False
    do_geo_eval: bool = True

    prev_schedule_path: Path = (
        path_settings.MEASUREMENTS_SCHEDULE
        / "meshed_cdns_pings_test__2a0cc0c3-cb91-42bb-bf8b-20ef887390ed.json"
    )

    if do_measurement:
        asyncio.run(meshed_ping_cdns(prev_schedule_path))
    if do_latency_eval:
        latency_eval()
    if do_geo_eval:
        geo_eval()


if __name__ == "__main__":
    main()
