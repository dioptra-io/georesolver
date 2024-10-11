import asyncio
import numpy as np

from tqdm import tqdm
from numpy import mean
from loguru import logger
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from geogiant.common.queries import (
    insert_scores,
    get_subnets,
    load_vp_subnets,
    get_ecs_results,
)
from geogiant.common.files_utils import load_csv
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings, ClickhouseSettings, setup_logger

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def score_jaccard(target_mapping: set, vp_mapping: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_mapping).intersection(set(vp_mapping))) / len(
        set(target_mapping).union(set(vp_mapping))
    )


async def score_filter(
    subnets: list[str],
    ecs_mapping_table: list[str],
    score_table: str,
    verbose: bool = False,
) -> list[str]:
    """filter and return subnets for which score calculation is yet to be done"""
    filtered_subnets = []

    # Check if ECS mapping exists for part of the subnets
    cached_ecs_subnets = get_subnets(
        table_name=ecs_mapping_table,
        print_error=verbose,
    )

    if not cached_ecs_subnets:
        return [], subnets

    cached_score_subnets = get_subnets(
        table_name=score_table,
        print_error=verbose,
    )

    no_score_subnet = set(subnets).difference(set(cached_score_subnets))
    filtered_subnets = set(no_score_subnet).intersection(set(cached_ecs_subnets))

    return list(filtered_subnets), list(no_score_subnet)


def score_parse(
    target_scores: dict,
    answer_granularity: str = "answer_subnets",
    metric: str = "jaccard",
) -> list[str]:
    """parse score function of granularity and return list for insert"""
    score_data = []
    for subnet, vps_subnet_score in target_scores.items():
        for vp_subnet, score in vps_subnet_score:
            score_data.append(
                f"{subnet},\
                {vp_subnet},\
                {metric},\
                {answer_granularity},\
                {score}"
            )

    return score_data


def score_per_vps_experimental(
    target_mapping: dict[dict],
    vps_mapping: dict,
) -> tuple[dict]:
    # maximum acheivable score (equal to the number of hostnames)
    maximum_score = len(target_mapping)

    # number of VPs to keep in memory
    best_vp_scores = np.zeros(50)

    vps_score = defaultdict(int)
    for vp_subnet in vps_mapping:
        nb_hostnames_done = 0
        for hostname, target_answers in target_mapping.items():

            # check if VP's subnet will bring better results
            if current_score := vps_score[vp_subnet]:
                best_acheivable_score = (
                    current_score + maximum_score - nb_hostnames_done
                )
                # no matter what, score will not be better
                if best_acheivable_score < best_vp_scores[0]:
                    break
            else:
                vps_score[vp_subnet] = 0

            try:
                vp_answers = vps_mapping[vp_subnet][hostname]
            except KeyError:
                continue

            # compute DNS similarity between target's redirection and vp's one
            vp_score = score_jaccard(target_answers, vp_answers)
            vps_score[vp_subnet] += vp_score
            nb_hostnames_done += 1

        # insert new vp into the best score memory, keep it sorted
        insert_index = np.searchsorted(best_vp_scores, vps_score[vp_subnet])
        best_vp_scores = np.insert(best_vp_scores, insert_index, vps_score[vp_subnet])
        best_vp_scores = np.delete(best_vp_scores, [0])

    return vps_score


def score_per_vps(
    target_mapping: dict[dict],
    vps_mapping: dict,
) -> tuple[dict]:
    """get vps score per hostname for a given target subnet"""
    vps_score = defaultdict(list)
    for hostname, target_answers in target_mapping.items():
        for vp_subnet in vps_mapping:
            try:
                vp_answers = vps_mapping[vp_subnet][hostname]
            except KeyError:
                continue

            vp_score = score_jaccard(target_answers, vp_answers)
            vps_score[vp_subnet].append(vp_score)

    return vps_score


def score_sort(vps_score: dict, nb_hostnames: int) -> dict[dict]:
    """calculate final target score"""
    return sorted(
        [
            (vp_subnet, hostname_scores / nb_hostnames)
            for vp_subnet, hostname_scores in vps_score.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )


def score_targets(args) -> None:
    """for each target, compute the ecs fingerprint similarity for each VP"""
    (
        target_subnets,
        vp_subnets,
        hostnames,
        targets_ecs_table,
        vps_ecs_table,
        output_logs,
    ) = args

    targets_mapping = get_ecs_results(
        dns_table=targets_ecs_table,
        subnets=target_subnets,
        hostname_filter=hostnames,
    )

    vps_mapping = get_ecs_results(
        dns_table=vps_ecs_table,
        subnets=vp_subnets,
        hostname_filter=hostnames,
    )

    if output_logs:
        output_file = open(output_logs, "a")
    else:
        output_file = None

    target_score = {}
    for target_subnet in tqdm(target_subnets, file=output_file):

        vps_score_subnet = score_per_vps_experimental(
            target_mapping=targets_mapping[target_subnet],
            vps_mapping=vps_mapping,
        )

        target_score[target_subnet] = score_sort(
            vps_score=vps_score_subnet, nb_hostnames=len(hostnames)
        )

    if output_file:
        output_file.close()

    return target_score


async def score_calculate(
    target_subnets: list[str],
    vp_subnets: list[str],
    hostnames: list[str],
    targets_ecs_table: str,
    vps_ecs_table: str,
    score_table: str,
    nb_cpu: int = None,
    output_logs: str = None,
) -> None:

    # batch of subnets on each CPUs
    nb_cpu = cpu_count() - 1 if not nb_cpu else nb_cpu
    batch_size = (len(target_subnets) // nb_cpu) + 1

    logger.info(f"Target subnets    :: {len(target_subnets)}")
    logger.info(f"VPs subnets       :: {len(vp_subnets)}")
    logger.info(f"Score hostnames   :: {len(hostnames)}")
    logger.info(f"Nb CPUs available :: {nb_cpu}")
    logger.info(f"Batch size        :: {batch_size}")

    if nb_cpu == 1:
        target_scores = score_targets(
            [
                target_subnets,
                vp_subnets,
                hostnames,
                targets_ecs_table,
                vps_ecs_table,
                output_logs,
            ]
        )
    else:

        target_scores = {}
        with Pool(nb_cpu) as pool:
            batches = []
            for i in range(0, len(target_subnets), batch_size):
                batches.append(
                    [
                        target_subnets[i : i + batch_size],
                        vp_subnets,
                        hostnames,
                        targets_ecs_table,
                        vps_ecs_table,
                        output_logs,
                    ]
                )

            batched_scores = pool.map(score_targets, batches)

            # unpack results
            for score in batched_scores:
                target_scores.update(score)

    score_data = score_parse(target_scores)

    await insert_scores(
        csv_data=score_data,
        output_table=score_table,
    )

    return target_scores


async def score_task(
    target_file: Path,
    hostname_file: Path,
    in_table: str,
    out_table: str,
    wait_time: int = 30,
    batch_size: int = 1_000,
    verbose: bool = False,
    log_path: Path = path_settings.LOG_PATH,
    output_logs: str = "score_task.log",
    nb_cpu: int = None,
    dry_run: bool = False,
) -> None:
    """run ecs mapping on batches of target subnets"""
    if output_logs and log_path:
        output_logs = log_path / output_logs
        setup_logger(output_logs)
    else:
        output_logs = None

    targets = load_csv(target_file, exit_on_failure=True)
    hostnames = load_csv(hostname_file, exit_on_failure=True)
    vp_subnets = load_vp_subnets(clickhouse_settings.VPS_FILTERED_FINAL_TABLE)
    subnets = list(set([get_prefix_from_ip(ip) for ip in targets]))

    while True:

        # get subnets on which score calculation is needed
        filtered_subnets, no_score_subnets = await score_filter(
            subnets=subnets,
            ecs_mapping_table=in_table,
            score_table=out_table,
            verbose=verbose,
        )

        logger.info(f"Original number of subnets          :: {len(subnets)}")
        logger.info(f"Remaining subnets for score process :: {len(no_score_subnets)}")
        logger.info(f"Subnets ready for score             :: {len(filtered_subnets)}")

        if not no_score_subnets:
            logger.info("Score calculation process completed")
            break

        if filtered_subnets:

            if dry_run:
                logger.info("Stopped Score process")
                break

            for i in range(0, len(filtered_subnets), batch_size):
                input_subnets = filtered_subnets[i : i + batch_size]
                logger.info(
                    f"Score:: batch={(i // batch_size) + 1}/{(len(filtered_subnets) // batch_size) + 1}"
                )

                await score_calculate(
                    target_subnets=input_subnets,
                    vp_subnets=vp_subnets,
                    hostnames=hostnames,
                    targets_ecs_table=in_table,
                    vps_ecs_table=clickhouse_settings.VPS_ECS_MAPPING_TABLE,
                    score_table=out_table,
                    nb_cpu=nb_cpu,
                    output_logs=output_logs,
                )

        else:
            logger.info("Waiting for ECS mapping to complete")
            await asyncio.sleep(wait_time)

            if dry_run:
                logger.info("Stopped Geolocation process")
                break


# profilin, testing, debugging
if __name__ == "__main__":

    targets = load_csv(path_settings.DATASET / "demo_targets.csv")
    subnets = [get_prefix_from_ip(addr) for addr in targets]
    hostnames = load_csv(path_settings.HOSTNAME_FILES / "hostnames_georesolver.csv")

    asyncio.run(
        score_task(
            subnets=subnets,
            hostnames=hostnames,
            ecs_mapping_table="demo_ecs_mapping",
            score_table="demo_score",
            batch_size=10,
            nb_cpu=1,
            output_logs=None,
        )
    )
