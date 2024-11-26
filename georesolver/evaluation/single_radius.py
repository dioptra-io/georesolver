"""this script aims at comparing the performances of georesolver against single-radius"""

import os
import httpx
import asyncio

from time import sleep
from pprint import pformat
from loguru import logger
from numpy import mean

from georesolver.agent import retrieve_pings
from georesolver.scheduler import create_agents
from georesolver.clickhouse.queries import get_pings_per_target, get_tables
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.common.files_utils import load_json, load_csv, dump_csv
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["RIPE_ATLAS_SECRET_KEY"] = (
    RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY_SECONDARY
)

NB_TARGETS = 100_000
SINGLE_RADIUS_TARGET_FILE = (
    path_settings.DATASET / "single_radius/single_radius_targets.csv"
)
SINGLE_RADIUS_MEASUREMENT_IDS = (
    path_settings.DATASET / "single_radius/single_radius_ids.csv"
)
SINGLE_RADIUS_PING_TABLE = "single_radius_ping"
GEORESOLVER_PING_TABLE = "single_radius_georesolver_ping"
GEORESOLVER_CONFIG_FILE = (
    path_settings.DATASET / "experiment_config/single_radius_config.json"
)


def get_single_radius_measurement_ids(wait_time: int = 30) -> None:
    """generate a file with single radius measurement ids"""
    tags = ["geolocation", "single-radius"]
    params = {"sort": ["-start_time"], "tags": tags, "status__in": "4", "af": 4}
    headers = {"Authorization": f"Key {RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY}"}

    logger.info("Retrieving measurement ids")
    logger.info(f"Number of measurement to retrive:: {NB_TARGETS}")

    measurement_ids = []
    with httpx.Client() as client:
        for _ in range(10):
            try:
                resp = client.get(
                    url=RIPEAtlasSettings().MEASUREMENT_URL,
                    params=params,
                    headers=headers,
                )
                resp = resp.json()
            except Exception:
                logger.info("Unsuported error, waiting")
                sleep(wait_time)
                continue

            if resp:
                break

        for measurement in resp["results"]:
            measurement_ids.append(measurement["id"])

        while len(measurement_ids) <= NB_TARGETS and resp["next"]:
            proggress = round((len(measurement_ids) / NB_TARGETS) * 100, 2)
            logger.info(
                f"Loading new page, {len(measurement_ids)} measurement retrieved {proggress}[%]"
            )
            for _ in range(10):
                try:
                    resp = client.get(url=resp["next"], headers=headers)
                    resp = resp.json()
                except Exception:
                    logger.info("Unsuported error, waiting")
                    sleep(wait_time)
                    continue
                if resp:
                    break
            for measurement in resp["results"]:
                measurement_ids.append(measurement["id"])

    dump_csv(measurement_ids, SINGLE_RADIUS_MEASUREMENT_IDS)


async def get_single_radius_measurement_results() -> None:
    """using previously retrieved measurement ids, get measurement results"""
    logger.info("Retrieving measurement results")

    measurement_ids = load_csv(SINGLE_RADIUS_MEASUREMENT_IDS)

    batch_size = 1_000
    for i in range(0, len(measurement_ids), batch_size):
        batch_ids = measurement_ids[i : i + batch_size]
        logger.info(f"batch:: {i} -> {i + batch_size}")
        await retrieve_pings(
            ids=batch_ids,
            output_table=SINGLE_RADIUS_PING_TABLE,
        )


def generate_target_file() -> None:
    """generate target file based on retrieved measurement results"""
    pings_per_target = get_pings_per_target(table_name=SINGLE_RADIUS_PING_TABLE)
    targets = [t for t in pings_per_target]
    dump_csv(targets, SINGLE_RADIUS_TARGET_FILE)


def load_dataset() -> None:
    """retrieve measurements using single radius tag"""
    if not SINGLE_RADIUS_MEASUREMENT_IDS.exists():
        get_single_radius_measurement_ids()

    if not SINGLE_RADIUS_PING_TABLE in get_tables():
        asyncio.run(get_single_radius_measurement_results())

    if not SINGLE_RADIUS_TARGET_FILE.exists():
        generate_target_file()


def run_georesolver() -> None:
    """perform georesolver measurement on single radius dataset"""
    config = load_json(GEORESOLVER_CONFIG_FILE)
    logger.info("Running GeoResolver with config:: ")
    logger.info("config=\n{}", pformat(config))

    agents = create_agents(GEORESOLVER_CONFIG_FILE)
    for agent in agents:
        agent.run()


def get_shortest_ping(ping_table: str) -> tuple[dict]:
    """retrieve georesolver shortest ping"""
    pings_per_target = get_pings_per_target(ping_table)

    shortest_ping_per_target = {}
    under_2_ms = {}
    for target, pings in pings_per_target.items():
        vp, min_rtt = min(pings, key=lambda x: x[-1])

        shortest_ping_per_target[target] = (vp, min_rtt)

        if min_rtt <= 2:
            under_2_ms[target] = (vp, min_rtt)

    return shortest_ping_per_target, under_2_ms


def shortest_ping_evaluation() -> None:
    """compute shortest ping for both single radius and georesolver, make figure"""
    shortest_ping_single_radius, under_2_ms_single_radius = get_shortest_ping(
        SINGLE_RADIUS_PING_TABLE
    )
    shortest_ping_georesolver, under_2_ms_georesolver = get_shortest_ping(
        GEORESOLVER_PING_TABLE
    )

    shortest_ping_single_radius = {
        key: val
        for key, val in shortest_ping_single_radius.items()
        if key in shortest_ping_georesolver
    }

    cdfs = []
    x, y = ecdf([min_rtt for _, min_rtt in shortest_ping_georesolver.values()])
    cdfs.append((x, y, "georesolver"))
    frac_under_2_ms_georesolver = get_proportion_under(x, y, 2)

    x, y = ecdf([min_rtt for _, min_rtt in shortest_ping_single_radius.values()])
    cdfs.append((x, y, "single-radius"))
    frac_under_2_ms_single_radius = get_proportion_under(x, y, 2)

    logger.info("Single radius")
    logger.info(f"Nb targets             :: {len(shortest_ping_single_radius)}")
    logger.info(f"Nb target under 2ms    :: {len(under_2_ms_single_radius)}")
    logger.info(f"Frac target under 2ms  :: {frac_under_2_ms_single_radius}")

    logger.info("GeoResolver")
    logger.info(f"Nb targets             :: {len(shortest_ping_georesolver)}")
    logger.info(f"Nb target under 2ms    :: {len(under_2_ms_georesolver)}")
    logger.info(f"Frac target under 2ms  :: {frac_under_2_ms_georesolver}")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="single_radius_vs_georesolver_shortest_ping",
        metric_evaluated="rtt",
        legend_pos="lower right",
    )


def cost_evaluation() -> None:
    """perform cost evaluation and plot CDF"""
    single_radius_pings = get_pings_per_target(SINGLE_RADIUS_PING_TABLE)

    cost_per_target = []
    for pings in single_radius_pings.values():
        cost_per_target.append(len(pings))

    cdfs = []
    avg_cost = mean(cost_per_target)
    cummulative_cost_single_radius = sum(cost_per_target)
    cummulative_cost_georesolver = 50 * len(cost_per_target)

    x, y = ecdf(cost_per_target)
    cdfs.append((x, y, "single-radius"))

    y = [50 for _ in range(len(x))]
    cdfs.append((x, y, "georesolver"))

    logger.info(f"Avg cost single-radius         :: {round(avg_cost)}")
    logger.info(f"Cummulative cost single-radius :: {cummulative_cost_single_radius}")
    logger.info(f"Cummulative cost GeoResolver   :: {cummulative_cost_georesolver}")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="single_radius_vs_georesolver_cost",
        metric_evaluated="",
        legend_pos="lower right",
        x_log_scale=False,
    )


def get_vps_asn_country() -> None:
    """return VPs asn and country"""
    pass


def asn_country_diversity_evaluation() -> None:
    """per target, get the number of country/AS represented by the VPs set"""
    pings_single_radius = get_pings_per_target(SINGLE_RADIUS_PING_TABLE)
    pings_georesolver = get_pings_per_target(GEORESOLVER_PING_TABLE)

    for target, pings in pings_georesolver.items():
        pass


def run_evaluation() -> None:
    """
    perform the evaluation:
        - latency comparison overall:
            - shortest ping single radius/georesolver
        - AS diversity/country diversity (CDF + mean per target)
        - VPs selection analysis (to define):
            - idea: cummulative distance VPs
            - idea: mean latency VPs
    """
    do_shortest_ping_evaluation: bool = True
    do_cost_evaluation: bool = True
    do_asn_country_diversity_evaluation: bool = True

    if do_shortest_ping_evaluation:
        shortest_ping_evaluation()

    if do_cost_evaluation:
        cost_evaluation()

    if do_asn_country_diversity_evaluation:
        asn_country_diversity_evaluation()


def main() -> None:
    """
    main function:
        - generate a dataset of single radius measurements
        - perform georesolver measurement on the dataset
        - evaluation
    """
    perform_measurement: bool = False
    make_eval: bool = True
    if perform_measurement:
        load_dataset()
        run_georesolver()

    if make_eval:
        run_evaluation()


if __name__ == "__main__":
    main()
