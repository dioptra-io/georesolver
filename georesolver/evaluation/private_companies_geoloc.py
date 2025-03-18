"""Test impact on latency of different prior geolocation knowledge for selecting VPs"""

import asyncio

from pathlib import Path
from loguru import logger
from random import sample
from datetime import datetime, timedelta

from georesolver.clickhouse.queries import (
    load_vps,
    get_measurement_ids,
    get_pings_per_target,
)
from georesolver.prober import RIPEAtlasProber, RIPEAtlasAPI
from georesolver.agent.insert_process import retrieve_pings
from georesolver.evaluation.evaluation_plot_functions import plot_multiple_cdf, ecdf
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_csv, dump_csv, load_json
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
ripe_atlas_settings = RIPEAtlasSettings()

PRIVATE_COMPANIES_PATH: Path = path_settings.DATASET / "private_companies_comparison"
TARGET_PATH: Path = PRIVATE_COMPANIES_PATH / "itdk_sample_targets.csv"
MEASURMENT_TAG = "meshed-cdns-pings-test"
OUTPUT_TABLE = "meshed_cdns_pings_test"


def load_sample_targets(
    input_path: Path, output_path: Path, sample_size: int = 10_000
) -> list[str]:
    """select a random sample from ITDK for measurements"""
    if not output_path.exists():
        itdk_targets = load_csv(input_path)
        sample_targets = sample(
            sample_targets,
            sample_size if sample_size < len(itdk_targets) else len(itdk_targets),
        )

        dump_csv(sample_targets, output_path)

        return sample_targets

    sample_targets = load_csv(output_path)
    return sample_targets


def get_schedule_estimated_geoloc(
    targets: list[str], vps: dict[dict], geoloc_per_target: Path
) -> list[tuple[str, int]]:
    """based on private companies estimated geoloc, select VPs"""
    measurement_schedule = []
    for target_addr in targets:
        target_geoloc = geoloc_per_target[target_addr]
        # find the 50 closest VPs
        vps_to_target_dst = []
        for vp in vps:
            dst_to_target = distance(
                vp["lat"], target_geoloc["lat"], vp["lon"], target_geoloc["lon"]
            )
            vps_to_target_dst.append((vp, dst_to_target))

        # order vps per shortest dst to target
        vps_to_target_dst = sorted(vps_to_target_dst, key=lambda x: x[-1])

        # select the first 50 VPs
        measurement_schedule.append(
            (target_addr, [vp["id"] for vp, _ in vps_to_target_dst[:50]])
        )

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


async def meshed_ping_cdns(measurement_schedule: list[tuple[str, int]]) -> None:
    """
    perform pings towards one IP address
    for each /24 prefix present in VPs ECS mapping redirection
    """
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


def latency_eval(input_tables: list[str]) -> None:
    """compare shortest ping obtained with each vp selection methodology"""
    cdfs = []
    for input_table in input_tables:
        logger.info(f"Retrieving pings for table:: {input_table}")
        pings_per_target = get_pings_per_target(input_table)

        # for each target, get shortest ping
        shortest_pings = []
        for target, pings in pings_per_target.items():
            shortest_ping = min(pings, key=lambda x: x[-1])

            shortest_pings.append(shortest_ping[-1])

        label = input_table.split("_")[0]
        x, y = ecdf(shortest_ping)
        cdfs.append((x, y, label))

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="private_companies_latency_comparison",
        metric_evaluated="rtt",
        x_log_scale=True,
    )


def main() -> None:
    """
    entry point:
        - run meshed measurements per CDNs (all VPs towards one IP addr per /24 answer)
        - evaluate redirection latency
        - evaluate max distance based on exact position + geolocation area of presence
    """
    do_maxmind_measurements: bool = True
    do_ip_info_measurements: bool = False
    do_latency_eval: bool = False

    sample_targets = load_sample_targets(
        path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv"
    )
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)

    if do_maxmind_measurements:
        maxmind_geoloc_per_target = load_json(
            path_settings.DATASET / "maxmind/geoloc_per_target.json"
        )
        measurement_schedule = get_schedule_estimated_geoloc(
            targets=sample_targets,
            vps=vps,
            geoloc_per_target=maxmind_geoloc_per_target,
        )
        asyncio.run(meshed_ping_cdns(measurement_schedule))

    if do_ip_info_measurements:
        ipinfo_geoloc_per_target = load_json(
            path_settings.DATASET / "ipinfo/geoloc_per_target.json"
        )
        measurement_schedule = get_schedule_estimated_geoloc(
            targets=sample_targets,
            vps=vps,
            geoloc_per_target=ipinfo_geoloc_per_target,
        )
        asyncio.run(meshed_ping_cdns(measurement_schedule))

    if do_latency_eval:
        latency_eval()


if __name__ == "__main__":
    main()
