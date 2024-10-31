"""ECS deprecation evaluation script: compare and evaluate how long before VPS ECS mapping become stale"""

import os

from loguru import logger
from pprint import pformat
from datetime import datetime
from collections import defaultdict

from georesolver.agent import ProcessNames
from georesolver.clickhouse.queries import load_targets, load_vps, load_target_geoloc
from georesolver.evaluation.evaluation_plot_functions import get_proportion_under, ecdf
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_json
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
os.environ["CLICKHOUSE_DATABASE"] = "GeoResolver_ecs_deprecation"
ch_settings = ClickhouseSettings()

EXPERIMENT_NAME = "ecs_test_deprecation"


def get_init_measurement(measurements: dict[dict]) -> None:
    """
    over all measurement, find the one that occured at day 0
    It has no round measurements and serve as a ref
    """
    init_measurement = {}
    round_measurements = {}
    for date, measurement in measurements.items():
        # first day: no run with new VPs mapping
        if len(measurement) == 1:
            init_measurement = measurement
        else:
            # only two measurement should have run each day
            # 1: with init vps mapping and 2: with new VPs mapping
            assert len(measurement) == 2
            round_measurements[date] = measurement

    return measurements, init_measurement


def load_measurements() -> None:
    """load measurements made on each days"""
    measurements = defaultdict(dict)

    for experiments_dir in path_settings.EXPERIMENT_PATH.iterdir():
        if EXPERIMENT_NAME in str(experiments_dir):
            config = load_json(experiments_dir / "ecs_deprecation_config.json")
            uuid = config["experiment_uuid"]

            start_time = config["start_time"].split(" ")[0]
            start_time = datetime.fromisoformat(start_time)

            # get VPs table, ping table
            for process in config["processes"]:
                if process["name"] == ProcessNames.SCORE_PROC.value:
                    vps_ecs_table = process["vps_ecs_table"]
                if process["name"] == ProcessNames.PING_PROC.value:
                    ping_table = process["out_table"]

            # check if init exp or round exp
            if "vps_init_ecs" in vps_ecs_table:
                exp_type = "init_vps_mapping"
            else:
                exp_type = "round_vps_mapping"

            measurements[start_time][exp_type] = {
                "uuid": uuid,
                "start_time": start_time,
                "ping_table": ping_table,
                "vps_ecs_table": vps_ecs_table,
            }

    measurements, init_measurement = get_init_measurement(measurements)

    return measurements, init_measurement


def get_geoloc_results(
    measurement_table: str, vps_per_addr: dict, target_per_addr: dict
) -> None:
    """return the number of IP addresses with a geolocation result under threshold"""
    geoloc = load_target_geoloc(measurement_table)
    d_error = []
    rtts = []
    for target_addr, geoloc in geoloc.items():
        try:
            vp_addr = geoloc[0]
            vp = vps_per_addr[vp_addr]
            vp_lat, vp_lon = vp["lat"], vp["lon"]
        except KeyError:
            logger.error(f"Cannot find geolocation for target addr:: {target_addr}")
            continue

        target = target_per_addr[target_addr]
        target_lat, target_lon = target["lat"], target["lon"]

        # distance errors
        d = distance(vp_lat, vp_lon, target_lat, target_lon)
        d_error.append(d)

        # min measured latency
        rtt = geoloc[-1]
        rtts.append(rtt)

    x, y = ecdf(d_error)
    under_40_km = get_proportion_under(x, y, threshold=40)

    x, y = ecdf(d_error)
    under_2_ms = get_proportion_under(x, y, threshold=2)

    return under_40_km, under_2_ms


def evaluation(measurements: dict[dict], init_measurement: dict[dict]) -> None:
    """
    compare the results between::
        1. new ping with init vps mapping vs. init pings
        2. new ping with round vps mapping vs. init pings
        3. new ping with round vps mapping vs. new ping with init vps mapping
    metric compared::
        1. proportion of IP addresses under 40km
        2. proportion of IP addresses under 2ms
    """
    init_results = []
    round_results = []
    ref_results = []
    # load data
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    targets = load_targets(ch_settings.VPS_FILTERED_FINAL_TABLE)

    # small parsing for fast vps retrieval
    vps_per_addr = {}
    for vp in vps:
        vps_per_addr[vp["addr"]] = vp
    target_per_addr = {}
    for target in targets:
        target_per_addr[target["addr"]] = target

    # get ref metrics (day 0)
    ref_measurement_table = init_measurement["init_vps_mapping"]["ping_table"]
    ref_under_40_km, ref_under_2_ms = get_geoloc_results(
        measurement_table=ref_measurement_table,
        vps_per_addr=vps_per_addr,
        target_per_addr=target_per_addr,
    )

    # get results for each round of measurements
    for start_time, measurement in measurements:
        init_mapping_measurement_table = measurement["init_vps_mapping"]
        round_mapping_measurement_table = measurement["round_vps_mapping"]

        init_under_40_km, init_under_2_ms = get_geoloc_results(
            measurement_table=init_mapping_measurement_table,
            vps_per_addr=vps_per_addr,
            target_per_addr=target_per_addr,
        )

        round_under_40_km, round_under_2_ms = get_geoloc_results(
            measurement_table=round_mapping_measurement_table,
            vps_per_addr=vps_per_addr,
            target_per_addr=target_per_addr,
        )

        init_results.append(start_time, (init_under_40_km, init_under_2_ms))
        round_results.append(start_time, (round_under_40_km, round_under_2_ms))
        ref_results.append(start_time, (ref_under_40_km, ref_under_2_ms))


def plot_results() -> None:
    """plot each results"""
    pass  # wip


def main() -> None:
    """entry point"""
    measurements, init_measurement = load_measurements()

    logger.info(f"Number of measurements:: {len(measurements)}")
    logger.info("{}", pformat(measurements))
    logger.info("Init measurement::")
    logger.info("{}", pformat(init_measurement))

    evaluation(measurements, init_measurement)


if __name__ == "__main__":
    main()
