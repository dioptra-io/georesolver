import time

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from numpy import mean
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from pych_client import ClickHouseClient

from geogiant.clickhouse import GetVPsSubnets, GetDNSMappingHostnames
from geogiant.common.files_utils import dump_pickle, load_json
from geogiant.common.settings import ClickhouseSettings, PathSettings
from geogiant.ecs_vp_selection.scores import TargetScores, get_scores

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

if __name__ == "__main__":
    hostname_file = "hostname_per_cdn_max_bgp_prefix.json"

    targets_table = clickhouse_settings.VPS_RAW
    vps_table = clickhouse_settings.VPS_RAW

    targets_ecs_table = "filtered_hostnames_ecs_mapping"
    vps_ecs_table = "filtered_hostnames_ecs_mapping"

    hostname_per_cdn = load_json(path_settings.DATASET / hostname_file)

    nb_hostnames = 10
    selected_hostnames_per_cdn = {}
    for cdn, hostnames in hostname_per_cdn.items():
        selected_hostnames_per_cdn[cdn] = hostname_per_cdn[cdn][:nb_hostnames]
        logger.info(f"{cdn=}, nb_hostnames={len(selected_hostnames_per_cdn[cdn])}")

    score_config = {
        "targets_table": targets_table,
        "vps_table": vps_table,
        "hostname_per_cdn": selected_hostnames_per_cdn,
        "targets_ecs_table": targets_ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": ["intersection", "jaccard"],
        "answer_granularities": ["answer_subnets"],
    }

    output_path = (
        path_settings.RESULTS_PATH
        / f"scores__all_cdns_10_hostname_per_cdn_{score_config['hostname_selection']}.pickle"
    )

    score = get_scores(score_config)

    dump_pickle(data=score, output_file=output_path)

    logger.info(f"Score calculation done, {output_path=}")
