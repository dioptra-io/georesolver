import asyncio

from loguru import logger
from pathlib import Path

from geogiant.scores import get_scores, TargetScores
from geogiant.common.files_utils import create_tmp_json_file
from geogiant.common.queries import get_subnets_mapping, insert_scores, get_subnets
from geogiant.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def parsed_score(score_per_granularity: dict, answer_granularity: str) -> list[str]:
    """parse score function of granularity and return list for insert"""
    score_data = []
    for subnet, score_per_metric in score_per_granularity.items():
        for metric, vps_subnet_score in score_per_metric.items():
            for vp_subnet, score in vps_subnet_score:
                score_data.append(
                    f"{subnet},\
                    {vp_subnet},\
                    {metric},\
                    {answer_granularity},\
                    {score}"
                )

    return score_data


async def calculate_scores(
    target_subnets: list[str],
    hostnames: list[str],
    ecs_mapping_table: str,
    score_table: str,
    output_logs: Path = None,
) -> None:
    """calculate score and insert"""

    subnet_tmp_path = create_tmp_json_file(target_subnets)

    score_config = {
        "targets_subnet_path": subnet_tmp_path,
        "vps_table": clickhouse_settings.VPS_FILTERED_TABLE,
        "selected_hostnames": hostnames,
        "targets_ecs_table": ecs_mapping_table,
        "vps_ecs_table": clickhouse_settings.VPS_ECS_MAPPING_TABLE,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": ["jaccard"],
        "answer_granularities": ["answer_subnets"],
        "output_logs": output_logs,
    }

    scores: TargetScores = get_scores(score_config)
    target_score_subnet = scores.score_answer_subnets

    score_data = parsed_score(target_score_subnet, "answer_subnets")

    await insert_scores(
        csv_data=score_data,
        output_table=score_table,
    )

    subnet_tmp_path.unlink()


async def filter_score_subnets(
    subnets: list[str],
    ecs_mapping_subnets: list[str],
    score_table: str,
) -> list[str]:
    """filter and return subnets for which score calculation is yet to be done"""
    filtered_subnets = []
    cached_subnets = get_subnets(table_name=score_table, subnets=subnets)
    no_score_subnet = set(subnets).difference(set(cached_subnets))
    filtered_subnets = set(no_score_subnet).intersection(set(ecs_mapping_subnets))

    return list(filtered_subnets), list(no_score_subnet)


async def score_task(
    subnets: list[str],
    hostnames: list[str],
    ecs_mapping_table: str,
    score_table: str,
    wait_time: int = 30,
    batch_size: int = 1_000,
    verbose: bool = False,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "score_task.log",
) -> None:
    """run ecs mapping on batches of target subnets"""
    setup_logger(log_path / output_logs)

    while True:

        # Check if ECS mapping exists for part of the subnets
        ecs_mapping_subnets = get_subnets_mapping(
            dns_table=ecs_mapping_table,
            subnets=subnets,
            print_error=verbose,
        )

        if ecs_mapping_subnets:

            # Check if some score were already calculated
            filtered_subnets, no_score_subnets = await filter_score_subnets(
                subnets,
                ecs_mapping_subnets,
                score_table,
            )

            logger.info(f"Original number of subnets:: {len(subnets)}")
            logger.info(
                f"Remaining subnets on which to calculate score:: {len(no_score_subnets)}"
            )

            if not no_score_subnets:
                logger.info("Score calculation process completed")
                break

            # ECS mapping
            if filtered_subnets:

                logger.info(f"Calculating score for:: {len(filtered_subnets)} subnets")

                for i in range(0, len(filtered_subnets), batch_size):
                    input_subnets = filtered_subnets[i : i + batch_size]
                    logger.info(
                        f"Score:: batch={(i // batch_size) + 1}{(len(filtered_subnets) // batch_size) + 1}"
                    )

                    await calculate_scores(
                        target_subnets=input_subnets,
                        hostnames=hostnames,
                        ecs_mapping_table=ecs_mapping_table,
                        score_table=score_table,
                        output_logs=log_path / output_logs,
                    )

            else:
                logger.info("Waiting for ECS mapping to complete")
                await asyncio.sleep(wait_time)

        else:
            logger.info("Waiting for ECS mapping to complete")
            await asyncio.sleep(wait_time)
