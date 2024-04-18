from multiprocessing import Pool, cpu_count
from numpy import mean
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from pathlib import Path

from geogiant.common.files_utils import dump_pickle, load_json, load_pickle
from geogiant.common.utils import TargetScores
from geogiant.common.queries import (
    get_subnets_mapping,
    load_target_subnets,
    load_vp_subnets,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def intersection_score(target_mapping: set, vp_mapping: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_mapping).intersection(set(vp_mapping))) / min(
        (len(set(target_mapping)), len(set(vp_mapping)))
    )


def intersection_scope_linear_weight_score(
    target_mapping: set, vp_mapping: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_mapping).intersection(set(vp_mapping)))
        / min((len(set(target_mapping)), len(set(vp_mapping))))
        * (target_source_scope + vp_source_scope)
    )


def intersection_scope_poly_weight_score(
    target_mapping: set, vp_mapping: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_mapping).intersection(set(vp_mapping)))
        / min((len(set(target_mapping)), len(set(vp_mapping))))
        * (target_source_scope**2 + vp_source_scope**2)
    )


def intersection_scope_exp_weight_score(
    target_mapping: set, vp_mapping: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_mapping).intersection(set(vp_mapping)))
        / min((len(set(target_mapping)), len(set(vp_mapping))))
        * (1 / 2 ** (24 - vp_source_scope) + 1 / 2 ** (24 - target_source_scope))
    )


def jaccard_score(target_mapping: set, vp_mapping: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_mapping).intersection(set(vp_mapping))) / len(
        set(target_mapping).union(set(vp_mapping))
    )


def jaccard_scope_linear_weight_score(
    target_mapping: set, vp_mapping: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_mapping).intersection(set(vp_mapping)))
        / len(set(target_mapping).union(set(vp_mapping)))
        * (target_source_scope + vp_source_scope)
    )


def jaccard_scope_poly_weight_score(
    target_mapping: set, vp_mapping: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_mapping).intersection(set(vp_mapping)))
        / len(set(target_mapping).union(set(vp_mapping)))
        * (target_source_scope**2 + vp_source_scope**2)
    )


def jaccard_scope_exp_weight_score(
    target_mapping: set, vp_mapping: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_mapping).intersection(set(vp_mapping)))
        / len(set(target_mapping).union(set(vp_mapping)))
        * (1 / 2 ** (24 - vp_source_scope) + 1 / 2 ** (24 - target_source_scope))
    )


def get_vp_score_per_metric(
    target_mapping: list,
    vp_subnet: str,
    vp_mapping: list,
    target_source_scope: int,
    vp_source_scope: int,
    vp_score_score_per_metric: dict[dict[list]],
    score_config: dict,
) -> dict[float]:
    """compute the score of the VP following different metrics"""

    if "intersection" in score_config["score_metric"]:
        score = intersection_score(target_mapping, vp_mapping)
        try:
            vp_score_score_per_metric["intersection"][vp_subnet].append(score)
        except KeyError:
            vp_score_score_per_metric["intersection"] = defaultdict(list)
            vp_score_score_per_metric["intersection"][vp_subnet].append(score)

    if "jaccard" in score_config["score_metric"]:
        score = jaccard_score(target_mapping, vp_mapping)

        try:
            vp_score_score_per_metric["jaccard"][vp_subnet].append(score)
        except KeyError:
            vp_score_score_per_metric["jaccard"] = defaultdict(list)
            vp_score_score_per_metric["jaccard"][vp_subnet].append(score)

    if "intersection_scope_linear_weight" in score_config["score_metric"]:
        score = intersection_scope_linear_weight_score(
            target_mapping, vp_mapping, target_source_scope, vp_source_scope
        )
        try:
            vp_score_score_per_metric["intersection_scope_linear_weight"][
                vp_subnet
            ].append(score)
        except KeyError:
            vp_score_score_per_metric["intersection_scope_linear_weight"] = defaultdict(
                list
            )
            vp_score_score_per_metric["intersection_scope_linear_weight"][
                vp_subnet
            ].append(score)

    if "intersection_scope_poly_weight" in score_config["score_metric"]:
        score = intersection_scope_poly_weight_score(
            target_mapping, vp_mapping, target_source_scope, vp_source_scope
        )
        try:
            vp_score_score_per_metric["intersection_scope_poly_weight"][
                vp_subnet
            ].append(score)
        except KeyError:
            vp_score_score_per_metric["intersection_scope_poly_weight"] = defaultdict(
                list
            )
            vp_score_score_per_metric["intersection_scope_poly_weight"][
                vp_subnet
            ].append(score)

    if "intersection_scope_exp_weight" in score_config["score_metric"]:
        score = intersection_scope_exp_weight_score(
            target_mapping, vp_mapping, target_source_scope, vp_source_scope
        )
        try:
            vp_score_score_per_metric["intersection_scope_exp_weight"][
                vp_subnet
            ].append(score)
        except KeyError:
            vp_score_score_per_metric["intersection_scope_exp_weight"] = defaultdict(
                list
            )
            vp_score_score_per_metric["intersection_scope_exp_weight"][
                vp_subnet
            ].append(score)

    # Jaccard score computation
    if "jaccard_scope_linear_weight" in score_config["score_metric"]:
        score = jaccard_scope_linear_weight_score(
            target_mapping, vp_mapping, target_source_scope, vp_source_scope
        )
        try:
            vp_score_score_per_metric["jaccard_scope_linear_weight"][vp_subnet].append(
                score
            )
        except KeyError:
            vp_score_score_per_metric["jaccard_scope_linear_weight"] = defaultdict(list)
            vp_score_score_per_metric["jaccard_scope_linear_weight"][vp_subnet].append(
                score
            )

    if "jaccard_scope_poly_weight" in score_config["score_metric"]:
        score = jaccard_scope_poly_weight_score(
            target_mapping, vp_mapping, target_source_scope, vp_source_scope
        )
        try:
            vp_score_score_per_metric["jaccard_scope_poly_weight"][vp_subnet].append(
                score
            )
        except KeyError:
            vp_score_score_per_metric["jaccard_scope_poly_weight"] = defaultdict(list)
            vp_score_score_per_metric["jaccard_scope_poly_weight"][vp_subnet].append(
                score
            )

    if "jaccard_scope_exp_weight" in score_config["score_metric"]:
        score = jaccard_scope_exp_weight_score(
            target_mapping, vp_mapping, target_source_scope, vp_source_scope
        )
        try:
            vp_score_score_per_metric["jaccard_scope_exp_weight"][vp_subnet].append(
                score
            )
        except KeyError:
            vp_score_score_per_metric["jaccard_scope_exp_weight"] = defaultdict(list)
            vp_score_score_per_metric["jaccard_scope_exp_weight"][vp_subnet].append(
                score
            )

    return vp_score_score_per_metric


def get_vps_score_per_hostname(
    target_mapping: dict[dict],
    vps_mapping: dict,
    score_config: dict,
) -> tuple[dict]:
    """get vps score per hostname for a given target subnet"""
    vp_score_answers = {}
    vp_score_answer_subnets = {}
    vp_score_bgp_prefixes = {}

    for hostname, target_mapping in target_mapping.items():
        for vp_subnet in vps_mapping:
            try:
                vp_mapping = vps_mapping[vp_subnet][hostname]
            except KeyError:
                continue

            if "answers" in score_config["answer_granularities"]:
                vp_score_answers = get_vp_score_per_metric(
                    target_mapping=target_mapping["answers"],
                    vp_subnet=vp_subnet,
                    vp_mapping=vp_mapping["answers"],
                    target_source_scope=target_mapping["source_scope"],
                    vp_source_scope=vp_mapping["source_scope"],
                    vp_score_score_per_metric=vp_score_answers,
                    score_config=score_config,
                )

            if "answer_subnets" in score_config["answer_granularities"]:
                vp_score_answer_subnets = get_vp_score_per_metric(
                    target_mapping=target_mapping["answer_subnets"],
                    vp_subnet=vp_subnet,
                    vp_mapping=vp_mapping["answer_subnets"],
                    target_source_scope=target_mapping["source_scope"],
                    vp_source_scope=vp_mapping["source_scope"],
                    vp_score_score_per_metric=vp_score_answer_subnets,
                    score_config=score_config,
                )

            if "answer_bgp_prefixes" in score_config["answer_granularities"]:
                vp_score_bgp_prefixes = get_vp_score_per_metric(
                    target_mapping=target_mapping["answer_bgp_prefixes"],
                    vp_subnet=vp_subnet,
                    vp_mapping=vp_mapping["answer_bgp_prefixes"],
                    target_source_scope=target_mapping["source_scope"],
                    vp_source_scope=vp_mapping["source_scope"],
                    vp_score_score_per_metric=vp_score_answers,
                    score_config=score_config,
                )

    return vp_score_answers, vp_score_answer_subnets, vp_score_bgp_prefixes


def get_sorted_score(vps_score: dict) -> dict[dict]:
    """calculate final target score based on vps score for each target"""
    score_per_metric = {}
    max_vp = 5_00
    for metric, vps_score in vps_score.items():
        score_per_metric[metric] = sorted(
            [
                (vp_subnet, mean(hostname_scores))
                for vp_subnet, hostname_scores in vps_score.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:max_vp]

    return score_per_metric


def get_hostname_score(args) -> None:
    """for each target, compute the ecs fingerprint similarity for each VP"""
    (
        target_subnets,
        vp_subnets,
        score_config,
    ) = args

    hostnames, _ = load_hostnames(score_config["hostname_per_cdn"])

    if target_mapping_path := score_config["targets_subnet_path"]:
        targets_mapping = load_pickle(path_settings.DATASET / "targets_subnet.json")
    else:
        targets_mapping = get_subnets_mapping(
            dns_table=score_config["targets_ecs_table"],
            subnets=[s for s in target_subnets],
            hostname_filter=hostnames,
        )

    if vps_mapping_path := score_config["vps_mapping_path"]:
        vps_mapping = load_pickle(Path(vps_mapping_path))
    else:
        vps_mapping = get_subnets_mapping(
            dns_table=score_config["vps_ecs_table"],
            subnets=[s for s in vp_subnets],
            hostname_filter=hostnames,
        )

    logger.debug(f"{len(targets_mapping)=}")

    target_score_answer = defaultdict(dict)
    target_score_subnet = defaultdict(dict)
    target_score_bgp_prefix = defaultdict(dict)
    for target_subnet in tqdm(target_subnets):

        vps_score_answer, vps_score_subnet, vps_score_bgp_prefix = (
            get_vps_score_per_hostname(
                target_mapping=targets_mapping[target_subnet],
                vps_mapping=vps_mapping,
                score_config=score_config,
            )
        )

        if "answers" in score_config["answer_granularities"]:
            target_score_answer[target_subnet] = get_sorted_score(
                vps_score=vps_score_answer,
            )

        if "answer_subnets" in score_config["answer_granularities"]:
            target_score_subnet[target_subnet] = get_sorted_score(
                vps_score=vps_score_subnet,
            )

        if "answer_bgp_prefixes" in score_config["answer_granularities"]:
            target_score_bgp_prefix[target_subnet] = get_sorted_score(
                vps_score=vps_score_bgp_prefix,
            )

    return target_score_answer, target_score_subnet, target_score_bgp_prefix


def load_hostnames(hostname_per_cdn: dict) -> list[str]:
    """hostname selection is a dict of hostname per organization, return the raw list of hostnames"""
    selected_hostnames = set()
    selected_cdns = set()
    for cdn, hostnames in hostname_per_cdn.items():
        selected_hostnames.update(hostnames)
        selected_cdns.add(cdn)

    return selected_hostnames, selected_cdns


def get_scores(score_config: dict) -> None:
    targets_table = score_config["targets_table"]
    vps_table = score_config["vps_table"]

    hostname_per_cdn = score_config["hostname_per_cdn"]
    hostnames, cdns = load_hostnames(hostname_per_cdn)

    if targets_subnet_path := score_config["targets_subnet_path"]:
        target_subnets = load_json(path_settings.DATASET / "targets_subnet.json")
    else:
        target_subnets = load_target_subnets(targets_table)

    if vps_subnet_path := score_config["vps_subnet_path"]:
        vp_subnets = load_json(path_settings.DATASET / "vps_subnet.json")
    else:
        vp_subnets = load_vp_subnets(vps_table)

    logger.info(f"score calculation with:: {len(hostnames)} hostnames")

    # avoid overloading cpu
    if len(hostnames) > 5_00:
        usable_cpu = cpu_count() - 1
    else:
        usable_cpu = cpu_count() - 1

    if usable_cpu < 1:
        usable_cpu = 1

    batch_size = len(target_subnets) // usable_cpu

    logger.info(f"Nb CPUs available:: {cpu_count()} (number of cpu used: {usable_cpu})")

    target_score_answer = {}
    target_score_subnet = {}
    target_score_bgp_prefix = {}
    with Pool(usable_cpu) as pool:
        if batch_size == 0:
            batch_size = 1

        logger.info(f"Calculating scores on batch of:: {batch_size} targets")

        batches = []
        for i in range(0, len(target_subnets), batch_size):
            batch_target_subnets = target_subnets[i : i + batch_size]

            batch = [
                batch_target_subnets,
                vp_subnets,
                score_config,
            ]
            batches.append(batch)

        logger.info(f"Running score calculation on:: {len(batches)} batches")

        batched_scores = pool.map(get_hostname_score, batches)

        for batch in batched_scores:
            batch_score_answer = batch[0]
            batch_score_subnet = batch[1]
            batch_score_bgp_prefix = batch[2]

            if batch_score_answer:
                for subnet, score_per_metric in batch_score_answer.items():
                    target_score_answer[subnet] = score_per_metric

            if batch_score_subnet:
                for subnet, score_per_metric in batch_score_subnet.items():
                    target_score_subnet[subnet] = score_per_metric

            if batch_score_bgp_prefix:
                for subnet, score_per_metric in batch_score_bgp_prefix.items():
                    target_score_bgp_prefix[subnet] = score_per_metric

    logger.info(f"{len(target_score_subnet)}")

    score = TargetScores(
        score_config=score_config,
        hostnames=hostnames,
        cdns=cdns,
        score_answers=target_score_answer,
        score_answer_subnets=target_score_subnet,
        score_answer_bgp_prefixes=target_score_bgp_prefix,
    )

    dump_pickle(data=score, output_file=score_config["output_path"])


if __name__ == "__main__":
    score_config = load_json(path_settings.DATASET / "score_config.json")
    get_scores(score_config)
