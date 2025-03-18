from multiprocessing import Pool, cpu_count
from numpy import mean
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from pathlib import Path

from geogiant.common.files_utils import dump_pickle, load_json
from geogiant.common.queries import (
    get_subnets_mapping,
    load_target_subnets,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def jaccard_score(target_mapping: set, vp_mapping: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_mapping).intersection(set(vp_mapping))) / len(
        set(target_mapping).union(set(vp_mapping))
    )


def get_vps_score_per_hostname(
    target_mapping: dict[dict],
    landmarks_mapping: dict,
) -> tuple[dict]:
    """get vps score per hostname for a given target subnet"""
    vp_score_per_hostname = defaultdict(list)
    for hostname, target_mapping in target_mapping.items():
        for subnet in landmarks_mapping:
            try:
                landmark_mapping = landmarks_mapping[subnet][hostname]
            except KeyError:
                continue

            score = jaccard_score(
                target_mapping["answer_subnets"], landmark_mapping["answer_subnets"]
            )
            vp_score_per_hostname[subnet].append(score)

    return vp_score_per_hostname


def get_sorted_score(vps_score: dict) -> dict[dict]:
    """calculate final target score based on vps score for each target"""
    score = sorted(
        [
            (vp_subnet, mean(hostname_scores))
            for vp_subnet, hostname_scores in vps_score.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return score


def get_hostname_score(args) -> None:
    """for each target, compute the ecs fingerprint similarity for each VP"""
    (
        score_schedule,
        hostnames,
        targets_ecs_table,
        vps_ecs_table,
        landmarks_ecs_table,
    ) = args

    # load targets mapping
    targets_mapping = get_subnets_mapping(
        dns_table=targets_ecs_table,
        subnets=[s["target_subnet"] for s in score_schedule],
        hostname_filter=hostnames,
    )

    # load vps ecs mapping
    vps_mapping = {}
    vps_subnets = []
    for target_schedule in score_schedule:
        vps_subnets.extend([vp_subnet for vp_subnet in target_schedule["vps_subnet"]])

    batch_size = 1_000
    for i in range(0, len(vps_subnets), batch_size):
        batch_mapping = get_subnets_mapping(
            dns_table=vps_ecs_table,
            subnets=vps_subnets[i : i + batch_size],
            hostname_filter=hostnames,
        )

        vps_mapping.update(batch_mapping)

    logger.info(f"VPs mapping:: {len(vps_mapping)} subnets")

    # load landmarks ecs mapping
    landmarks_subnets = []
    for target_schedule in score_schedule:
        landmarks_subnets.extend(
            [landmark for landmark in target_schedule["landmarks_subnet"]]
        )

    logger.info(f"Landmarks subnets:: {len(landmarks_subnets)} subnets")
    landmarks_mapping = {}
    for i in range(0, len(vps_subnets), batch_size):
        batch_mapping = get_subnets_mapping(
            dns_table=landmarks_ecs_table,
            subnets=landmarks_subnets[i : i + batch_size],
            hostname_filter=hostnames,
        )
        landmarks_mapping.update(batch_mapping)

    logger.info(f"Landmarks mapping:: {len(landmarks_mapping)} subnets")

    # merge all mapping so everything is within the same data structure
    landmarks_mapping.update(vps_mapping)

    logger.debug(f"Total target subnets subnets:: {len(targets_mapping)=}")

    target_score_subnet = defaultdict(dict)
    for target_schedule in tqdm(score_schedule):
        target_subnet = target_schedule["target_subnet"]
        target_mapping = targets_mapping[target_subnet]

        # extract target specific landmark mapping
        target_landmarks_mapping = {}
        target_vps_mapping = {}
        for subnet in landmarks_mapping:
            if subnet in target_schedule["vps_subnet"]:
                target_vps_mapping[subnet] = vps_mapping[subnet]
            if subnet in target_schedule["landmarks_subnet"]:
                target_landmarks_mapping[subnet] = landmarks_mapping[subnet]

        logger.info(f"{target_subnet=}:: {len(target_landmarks_mapping)} landmarks")

        vps_score = get_vps_score_per_hostname(
            target_mapping=target_mapping,
            landmarks_mapping=target_vps_mapping,
        )

        landmarks_score = get_vps_score_per_hostname(
            target_mapping=target_mapping,
            landmarks_mapping=target_landmarks_mapping,
        )

        target_score_subnet[target_subnet]["vps_score"] = get_sorted_score(
            vps_score=vps_score,
        )

        target_score_subnet[target_subnet]["landmarks_score"] = get_sorted_score(
            vps_score=landmarks_score,
        )

    return target_score_subnet


def get_scores(
    hostnames: list[str],
    score_schedule: list[dict],
    target_ecs_table: str,
    vps_ecs_table: str,
    landmarks_ecs_table: str,
    output_path: Path,
) -> None:

    logger.info(f"score calculation:: {len(hostnames)} hostnames")
    logger.info(f"score calculation:: {len(score_schedule)} target subnets")

    # avoid overloading cpu
    if len(hostnames) > 5_00:
        usable_cpu = cpu_count() - 1
    else:
        usable_cpu = cpu_count() - 1

    if usable_cpu < 1:
        usable_cpu = 1

    # usable_cpu = 1

    batch_size = len(score_schedule) // usable_cpu + 1

    logger.info(f"Nb CPUs available:: {cpu_count()} (number of cpu used: {usable_cpu})")

    target_score = {}
    with Pool(usable_cpu) as pool:
        if batch_size == 0:
            batch_size = 1

        logger.info(f"Calculating scores on batch of:: {batch_size} targets")

        batches = []
        for i in range(0, len(score_schedule), batch_size):
            batch = score_schedule[i : i + batch_size]
            batches.append(
                (
                    batch,
                    hostnames,
                    target_ecs_table,
                    vps_ecs_table,
                    landmarks_ecs_table,
                )
            )

        logger.info(f"Running score calculation on:: {len(batches)} batches")

        batched_scores = pool.map(get_hostname_score, batches)

        for batch in batched_scores:
            batch_score = batch

            for subnet, scores in batch_score.items():
                target_score[subnet] = scores

    logger.info(f"{len(target_score)}")

    dump_pickle(data=target_score, output_file=Path(output_path))

    return scores
