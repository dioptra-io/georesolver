import asyncio
import httpx
import time

from dateutil import parser
from pathlib import Path
from datetime import datetime, timedelta
from pyasn import pyasn
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pych_client import ClickHouseClient


from geogiant.clickhouse import GetDstPrefix
from geogiant.prober.ripe_api import RIPEAtlasAPI
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.evaluation.ecs_geoloc_eval import get_ecs_vps
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.queries import get_min_rtt_per_vp, get_pings_per_target, load_vps
from geogiant.common.utils import TargetScores, get_parsed_vps
from geogiant.common.files_utils import (
    load_csv,
    load_json,
    dump_json,
    load_pickle,
    load_anycatch_data,
    decompress,
    load_iter,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# CLICKHOUSE TABLE
ping_ripe_altas_table = "pings_ripe_altas"
ping_ripe_ip_map_table = "pings_ripe_ip_map_from_tag"
ecs_table = "ripe_ip_map_mapping_ecs"
vps_ecs_table = "vps_mapping_ecs"

# FILE PATHS
ripe_ip_map_subnets_path = path_settings.DATASET / "ripe_ip_map_subnets.json"
measurement_ids_path = path_settings.DATASET / "ripe_ip_map_ids.json"
filtered_measurement_ids_path = path_settings.DATASET / "ripe_ip_map_filtered_ids.json"
vps_subnet_path = path_settings.DATASET / "vps_subnet.json"
score_file = (
    path_settings.RESULTS_PATH
    / f"ripe_ip_map_evaluation/scores__best_hostname_geo_score.pickle"
)
results_file = (
    path_settings.RESULTS_PATH
    / f"ripe_ip_map_evaluation/{'results' + str(score_file).split('scores')[-1]}"
)

# CONSTANT PARAMETERS
latency_threshold = 4
probing_budget = 500

START_DATE: str = "2024-04-17"
END_DATE: str = "2024-04-18"


def get_ripe_ip_map_ids(tag: str = "single-radius") -> None:
    """get all measurements from RIPE IP map single radius engine"""
    start_time = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_time = datetime.strptime(END_DATE, "%Y-%m-%d")

    start_time = int(start_time.timestamp())
    end_time = int(end_time.timestamp())

    logger.info(f"Fetching measurement metadata from:: {START_DATE} to {END_DATE}")

    params = {
        "tags": tag,
        "type": "ping",
        "af": 4,
        "start_time__gte": start_time,
        "sort": ["-msm_id"],
    }
    return RIPEAtlasAPI().get_measurements_from_tag(
        params=params, output_table=ping_ripe_ip_map_table
    )


async def get_ripe_ip_map_results() -> None:
    measurement_ids = get_ripe_ip_map_ids()


async def main() -> None:
    retrieve_public_measurements = True

    if retrieve_public_measurements:
        await get_ripe_ip_map_ids()


if __name__ == "__main__":
    asyncio.run(main())
