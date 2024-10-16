import asyncio

from uuid import uuid4
from collections import defaultdict
from loguru import logger

from geogiant.evaluation.scores import get_scores
from geogiant.common.queries import get_subnets
from geogiant.common.utils import TargetScores
from geogiant.common.files_utils import (
    load_json,
    create_tmp_json_file,
    load_pickle,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# CLICKHOUSE TABLE
PING_INTERNET_SCALE = "pings_routers"
ECS_TABLE = "internet_scale_mapping_ecs"
VPS_ECS_MAPPING_TABLE = "vps_ecs_mapping"

ALL_INTERNET_SCALE_SUBNETS_PATH = (
    path_settings.INTERNET_SCALE_DATASET / "all_internet_scale_subnets.json"
)
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnets_filtered.json"

# CONSTANT PARAMETERS
probing_budget = 50


def load_hostnames() -> dict:
    selected_hostnames_per_cdn_per_ns = load_json(
        path_settings.DATASET
        / f"hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json"
    )

    selected_hostnames = set()
    selected_hostnames_per_cdn = defaultdict(list)
    for ns in selected_hostnames_per_cdn_per_ns:
        for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
            selected_hostnames.update(hostnames)
            selected_hostnames_per_cdn[org].extend(hostnames)

    logger.info(f"{len(selected_hostnames)=}")

    return selected_hostnames_per_cdn, selected_hostnames


def load_subnets() -> list[str]:
    ecs_subnets = get_subnets(ECS_TABLE)

    logger.info(f"Number of resolved hostnames:: {len(ecs_subnets)}")

    cached_subnets = set()

    try:
        for file in path_settings.INTERNET_SCALE_RESULTS.iterdir():
            if "score__" in file.name:
                cached_score: TargetScores = load_pickle(file)
                cached_subnets.update(
                    [subnet for subnet in cached_score.score_answer_subnets]
                )
    except FileNotFoundError:
        logger.info("No subnets score cached")
        pass

    score_subnets = list(set(ecs_subnets).difference(cached_subnets))

    logger.info(
        f"Number of hostnames on which to calculate scores:: {len(score_subnets)}"
    )

    return score_subnets


async def main() -> None:

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
            "vps_ecs_table": VPS_ECS_MAPPING_TABLE,
            "hostname_selection": "max_bgp_prefix",
            "score_metric": ["jaccard"],
            "answer_granularities": ["answer_subnets"],
            "output_path": output_path,
        }

        get_scores(score_config)

        subnet_tmp_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
