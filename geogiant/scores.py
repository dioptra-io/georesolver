from tqdm import tqdm
from numpy import mean
from loguru import logger
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from geogiant.common.queries import get_ecs_results


def jaccard_score(target_mapping: set, vp_mapping: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_mapping).intersection(set(vp_mapping))) / len(
        set(target_mapping).union(set(vp_mapping))
    )


def get_vps_score_per_hostname(
    target_mapping: dict[dict],
    vps_mapping: dict,
) -> tuple[dict]:
    """get vps score per hostname for a given target subnet"""
    vps_score = defaultdict(list)
    # TODO: add breakpoint somewhere in this function
    for hostname, target_mapping in target_mapping.items():
        for vp_subnet in vps_mapping:
            try:
                vp_mapping = vps_mapping[vp_subnet][hostname]
            except KeyError:
                continue

            vp_score = jaccard_score(target_mapping, vp_mapping)
            vps_score[vp_subnet].append(vp_score)

    return vps_score


def get_sorted_score(vps_score: dict) -> dict[dict]:
    """calculate final target score"""
    return sorted(
        [
            (vp_subnet, mean(hostname_scores))
            for vp_subnet, hostname_scores in vps_score.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )


def get_hostname_score(args) -> None:
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

        vps_score_subnet = get_vps_score_per_hostname(
            target_mapping=targets_mapping[target_subnet],
            vps_mapping=vps_mapping,
        )

        target_score[target_subnet] = get_sorted_score(
            vps_score=vps_score_subnet,
        )

    if output_file:
        output_file.close()

    return target_score


def get_scores(
    target_subnets: list[str],
    vp_subnets: list[str],
    hostnames: list[str],
    targets_ecs_table: str,
    vps_ecs_table: str,
    output_logs: str = None,
) -> None:

    batch_size = (len(target_subnets) // cpu_count()) + 1

    logger.info(f"Target subnets    :: {len(target_subnets)}")
    logger.info(f"VPs subnets       :: {len(vp_subnets)}")
    logger.info(f"Score hostnames   :: {len(hostnames)}")
    logger.info(f"Nb CPUs available :: {cpu_count()}")
    logger.info(f"Batch size        :: {batch_size}")

    target_scores = {}
    with Pool(cpu_count()) as pool:
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

        batched_scores = pool.map(get_hostname_score, batches)

        # unpack results
        for score in batched_scores:
            target_scores.update(score)

    return target_scores
