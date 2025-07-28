"""Test impact on latency of different prior geolocation knowledge for selecting VPs"""

import asyncio
import maxminddb

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from random import sample
from datetime import datetime, timedelta

from georesolver.clickhouse.queries import (
    load_vps,
    get_targets,
    get_measurement_ids,
    get_pings_per_target,
)
from georesolver.prober import RIPEAtlasProber, RIPEAtlasAPI
from georesolver.agent.insert_process import retrieve_pings
from georesolver.evaluation.evaluation_plot_functions import (
    plot_multiple_cdf,
    ecdf,
    get_proportion_under,
)
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_csv, dump_csv
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


def load_sample_targets(
    input_path: Path, output_path: Path, sample_size: int = 15_000
) -> list[str]:
    """select a random sample from ITDK for measurements"""
    if not output_path.exists():
        itdk_targets = load_csv(input_path)
        sample_targets = sample(
            itdk_targets,
            sample_size if sample_size < len(itdk_targets) else len(itdk_targets),
        )

        dump_csv(sample_targets, output_path)

        logger.info(f"Itdk sampling done, {len(sample_targets)=}")

        return sample_targets

    sample_targets = load_csv(output_path)
    return sample_targets


def get_geoloc_per_target(targets: list[str], geoloc_db_path: Path) -> dict[dict]:
    """get geolocation per target fct private db"""
    geoloc_per_target = {}
    with maxminddb.open_database(geoloc_db_path) as r:
        for target in targets:
            geodb_response = r.get(target)
            if "ipinfo" in geoloc_db_path.name.lower():
                if "lat" not in geodb_response:
                    continue
                lat, lon = float(geodb_response["lat"]), float(geodb_response["lng"])
            if "mm" in geoloc_db_path.name.lower():
                if "latitude" not in geodb_response or not geodb_response["latitude"]:
                    continue
                lat, lon = float(geodb_response["latitude"]), float(
                    geodb_response["longitude"]
                )

            geoloc_per_target[target] = {
                "lat": lat,
                "lon": lon,
            }
    logger.info(f"Found geolocation for {len(geoloc_per_target)} targets")

    return geoloc_per_target


def get_measurement_schedule(
    geoloc_per_target: dict[dict],
    vps: dict[dict],
) -> list[tuple[str, int]]:
    """based on private companies estimated geoloc, select VPs"""
    measurement_schedule = []
    for target_addr, target_geoloc in tqdm(geoloc_per_target.items()):
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

    logger.info(f"Total number of measurements scheduled: {len(measurement_schedule)}")

    return measurement_schedule


async def insert_measurements(
    measurement_schedule: list[tuple],
    probing_tags: list[str],
    output_table: str,
    wait_time: int = 60,
    only_once: bool = False,
) -> None:
    """insert measurement once they are tagged as Finished on RIPE Atlas"""
    current_time = datetime.timestamp(datetime.now() - timedelta(days=7))
    cached_measurement_ids = set()

    logger.info("Starting inserting measurements")

    while True:
        # load measurement finished from RIPE Atlas
        stopped_measurement_ids = await RIPEAtlasAPI().get_stopped_measurement_ids(
            start_time=current_time, tags=probing_tags
        )

        # load already inserted measurement ids
        inserted_ids = get_measurement_ids(output_table)

        # stop measurement once all measurement are inserted
        all_measurement_ids = set(inserted_ids).union(cached_measurement_ids)
        # if len(all_measurement_ids) >= len(measurement_schedule):
        #     logger.info(
        #         f"All measurement inserted:: {len(inserted_ids)=}; {len(measurement_schedule)=}"
        #     )
        #     break

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
        batch_size = 1_000
        for i in range(0, len(measurement_to_insert), batch_size):

            logger.info(
                f"Batch {i // batch_size}/{len(measurement_to_insert) // batch_size}"
            )
            batch_measurement_ids = list(measurement_to_insert)[i : i + batch_size]

            await retrieve_pings(batch_measurement_ids, output_table, step_size=3)

        cached_measurement_ids.update(measurement_to_insert)
        current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

        if only_once:
            break

        await asyncio.sleep(wait_time)


async def run_measurement(
    measurement_schedule: list[tuple[str, int]],
    measurement_tag: str,
    output_table: str,
    check_cache: bool = False,
) -> None:
    """
    perform pings towards one IP address
    for each /24 prefix present in VPs ECS mapping redirection
    """
    # in case measurement failed previously
    if check_cache:
        # retrieve previously ongoing measurements
        await insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", measurement_tag],
            output_table=output_table,
            only_once=True,
        )

        logger.info("Filtering measurement schedule")
        logger.info(f"Original schedule: {len(measurement_schedule)} targets")

        # filter measurement schedule
        targets = get_targets(output_table)
        filtered_measurement_schedule = []
        for target_addr, vp_ids in tqdm(measurement_schedule):
            if target_addr in targets:
                continue

            filtered_measurement_schedule.append((target_addr, vp_ids))

        logger.info(f"Filtered schedule: {len(measurement_schedule)} targets")

        measurement_schedule = filtered_measurement_schedule

    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=measurement_tag,
        output_table=output_table,
        protocol="ICMP",
    )

    await asyncio.gather(
        prober.main(measurement_schedule),
        insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", measurement_tag],
            output_table=output_table,
        ),
    )

    logger.info("Meshed CDNs pings measurement done")


def evaluation(input_tables: list[tuple[str, str]]) -> None:
    """compare shortest ping obtained with each vp selection methodology"""
    cdfs = []
    for input_table, label in input_tables:
        under_2ms = []
        logger.info(f"Retrieving pings for table:: {input_table}")
        pings_per_target = get_pings_per_target(input_table)

        # for each target, get shortest ping
        shortest_pings = []
        for _, pings in pings_per_target.items():
            _, shortest_ping_rtt = min(pings, key=lambda x: x[-1])

            shortest_pings.append(shortest_ping_rtt)
            if shortest_ping_rtt < 2:
                under_2ms.append(shortest_ping_rtt)

        x, y = ecdf(shortest_pings)
        cdfs.append((x, y, label))

        frac_under2ms = get_proportion_under(x, y, 2)
        logger.info()
        logger.info("##################################################")
        logger.info(f"{input_table=}; {len(under_2ms)=}; {frac_under2ms=}")
        logger.info("##################################################")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="figure_4_right_georesolver_vp_selection_vs_private_db",
        metric_evaluated="rtt",
        x_log_scale=True,
        legend_pos="lower right",
    )


def main(
    do_maxmind_measurements: bool = False,
    do_ip_info_measurements: bool = False,
    do_evaluation: bool = True,
) -> None:
    """
    entry point:
        - run meshed measurements per CDNs (all VPs towards one IP addr per /24 answer)
        - !!! WE DO NOT RECOMMEND RUNNING MESHED MEASUREMENTS ON ALL CDNs IP ADDRS !!!
        - evaluate redirection latency
        - evaluate max distance based on exact position + geolocation area of presence
    """
    sample_targets = load_sample_targets(
        path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv",
        output_path=path_settings.DATASET / "itdk/random_sample_private_db.csv",
    )
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)

    if do_maxmind_measurements:
        geoloc_per_target = get_geoloc_per_target(
            sample_targets,
            path_settings.DATASET / "geoloc_db/mm_paid_output_2025-04-15.mmdb",
        )
        measurement_schedule = get_measurement_schedule(
            geoloc_per_target=geoloc_per_target,
            vps=vps,
        )
        asyncio.run(
            run_measurement(
                measurement_schedule,
                "maxmind-closest-vp-seed",
                "maxmind_closest_vp_seed_pings",
            )
        )

    if do_ip_info_measurements:
        geoloc_per_target = get_geoloc_per_target(
            sample_targets,
            path_settings.DATASET / "geoloc_db/ipinfo_2025-04-10.snapshot",
        )
        measurement_schedule = get_measurement_schedule(
            geoloc_per_target=geoloc_per_target,
            vps=vps,
        )
        asyncio.run(
            run_measurement(
                measurement_schedule,
                "ipinfo-closest-vp-seed-new",
                "ipinfo_closest_vp_seed_pings",
                check_cache=True,
            )
        )

    if do_evaluation:
        evaluation(
            [
                ("ipinfo_closest_vp_seed_pings", "IPinfo (50 closest VPs)"),
                ("georesolver_itdk_sample_ping", "GeoResolver"),
                ("maxmind_closest_vp_seed_pings", "maxmind (50 closest VPs)"),
            ]
        )


if __name__ == "__main__":
    main()
