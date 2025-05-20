"""calculate targets and vps ECS-DNS redirection similarity"""

from tqdm import tqdm
from numpy import mean
from pathlib import Path
from loguru import logger
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from georesolver.clickhouse.queries import get_subnets_mapping, get_min_rtt_per_vp
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_pickle, dump_pickle
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


def intersection_score(target_answers: set, vp_answers: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_answers).intersection(set(vp_answers))) / min(
        (len(set(target_answers)), len(set(vp_answers)))
    )


def intersection_scope_linear_weight_score(
    target_answers: set, vp_answers: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_answers).intersection(set(vp_answers)))
        / min((len(set(target_answers)), len(set(vp_answers))))
        * (target_source_scope + vp_source_scope)
    )


def intersection_scope_poly_weight_score(
    target_answers: set, vp_answers: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_answers).intersection(set(vp_answers)))
        / min((len(set(target_answers)), len(set(vp_answers))))
        * (target_source_scope**2 + vp_source_scope**2)
    )


def intersection_scope_exp_weight_score(
    target_answers: set, vp_answers: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_answers).intersection(set(vp_answers)))
        / min((len(set(target_answers)), len(set(vp_answers))))
        * (1 / 2 ** (24 - vp_source_scope) + 1 / 2 ** (24 - target_source_scope))
    )


def jaccard_score(target_answers: set, vp_answers: set) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return len(set(target_answers).intersection(set(vp_answers))) / len(
        set(target_answers).union(set(vp_answers))
    )


def jaccard_scope_linear_weight_score(
    target_answers: set, vp_answers: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_answers).intersection(set(vp_answers)))
        / len(set(target_answers).union(set(vp_answers)))
        * (target_source_scope + vp_source_scope)
    )


def jaccard_scope_poly_weight_score(
    target_answers: set, vp_answers: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_answers).intersection(set(vp_answers)))
        / len(set(target_answers).union(set(vp_answers)))
        * (target_source_scope**2 + vp_source_scope**2)
    )


def jaccard_scope_exp_weight_score(
    target_answers: set, vp_answers: set, target_source_scope: int, vp_source_scope: int
) -> float:
    """calculate the score similarity between a target subnet and a vp subnet"""
    return (
        len(set(target_answers).intersection(set(vp_answers)))
        / len(set(target_answers).union(set(vp_answers)))
        * (1 / 2 ** (24 - vp_source_scope) + 1 / 2 ** (24 - target_source_scope))
    )


def get_vps_score_per_hostname(
    target_mapping: dict[dict],
    vps_mapping: dict,
    score_metric: str,
    answer_granularity: str,
) -> tuple[dict]:
    """get vps score per hostname for a given target subnet"""
    vp_scores = defaultdict(list)
    for hostname, target_mapping in target_mapping.items():

        target_source_scope = target_mapping["source_scope"]
        target_answers = target_mapping[answer_granularity]

        for vp_subnet in vps_mapping:
            try:
                vp_mapping = vps_mapping[vp_subnet][hostname]
            except KeyError:
                continue

            vp_answers = vp_mapping[answer_granularity]
            vp_source_scope = vp_mapping["source_scope"]

            match score_metric:
                case "intersection":
                    score = intersection_score(target_answers, vp_answers)
                case "jaccard":
                    score = jaccard_score(target_answers, vp_answers)
                case "intersection_scope_linear_weight":
                    score = intersection_scope_linear_weight_score(
                        target_answers, vp_answers, target_source_scope, vp_source_scope
                    )
                case "intersection_scope_poly_weight":
                    score = intersection_scope_poly_weight_score(
                        target_answers, vp_answers, target_source_scope, vp_source_scope
                    )
                case "intersection_scope_exp_weight":
                    score = intersection_scope_exp_weight_score(
                        target_answers, vp_answers, target_source_scope, vp_source_scope
                    )
                case "jaccard_scope_linear_weight":
                    score = jaccard_scope_linear_weight_score(
                        target_answers, vp_answers, target_source_scope, vp_source_scope
                    )
                case "jaccard_scope_poly_weight":
                    score = jaccard_scope_poly_weight_score(
                        target_answers, vp_answers, target_source_scope, vp_source_scope
                    )
                case "jaccard_scope_exp_weight":
                    score = jaccard_scope_exp_weight_score(
                        target_answers, vp_answers, target_source_scope, vp_source_scope
                    )

            vp_scores[vp_subnet].append(score)

    return vp_scores


def get_sorted_score(vps_score: dict) -> dict[dict]:
    """calculate final target score based on vps score for each target"""
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
        hostnames,
        target_subnets,
        vp_subnets,
        target_ecs_table,
        vps_ecs_table,
        score_metric,
        answer_granularity,
        ipv6,
    ) = args

    targets_mapping = get_subnets_mapping(
        dns_table=target_ecs_table,
        subnets=target_subnets,
        hostname_filter=hostnames,
        ipv6=ipv6,
    )

    vps_mapping = get_subnets_mapping(
        dns_table=vps_ecs_table,
        subnets=vp_subnets,
        hostname_filter=hostnames,
        ipv6=ipv6,
    )

    logger.debug(f"Total subnets:: {len(targets_mapping)=}")

    target_scores = defaultdict(dict)
    for target_subnet in tqdm(target_subnets):

        vps_score = get_vps_score_per_hostname(
            target_mapping=targets_mapping[target_subnet],
            vps_mapping=vps_mapping,
            score_metric=score_metric,
            answer_granularity=answer_granularity,
        )

        target_scores[target_subnet] = get_sorted_score(
            vps_score=vps_score,
        )

    return target_scores


def load_hostnames(hostname_per_cdn: dict) -> list[str]:
    """hostname selection is a dict of hostname per organization, return the raw list of hostnames"""
    selected_hostnames = set()
    selected_cdns = set()
    for cdn, hostnames in hostname_per_cdn.items():
        selected_hostnames.update(hostnames)
        selected_cdns.add(cdn)

    return selected_hostnames, selected_cdns


def get_scores(
    output_path: Path,
    hostnames: list[str],
    target_subnets: list[str],
    vp_subnets: list[str],
    target_ecs_table: str,
    vps_ecs_table: str,
    answer_granularity: str = "answer_subnets",
    score_metric: str = "jaccard",
    ipv6: bool = False,
) -> None:

    # load cache score
    score_per_target_subnet = {}
    if output_path.exists():
        score_per_target_subnet = load_pickle(output_path)
        return score_per_target_subnet

    logger.info(f"score calculation:: {len(hostnames)} hostnames")
    logger.info(f"{len(target_subnets)=}")
    logger.info(f"{len(vp_subnets)=}")

    # spread computation on multiple cpus
    usable_cpu = cpu_count() - 2
    batch_size = len(target_subnets) // usable_cpu + 1

    logger.info(f"Nb CPUs available:: {cpu_count()} (number of cpu used: {usable_cpu})")

    # perform score computation in parrallel
    with Pool(usable_cpu) as pool:
        if batch_size == 0:
            batch_size = 1

        logger.info(f"Calculating scores on batch of:: {batch_size} targets")

        batches = []
        for i in range(0, len(target_subnets), batch_size):
            batch_target_subnets = target_subnets[i : i + batch_size]

            batch = [
                hostnames,
                batch_target_subnets,
                vp_subnets,
                target_ecs_table,
                vps_ecs_table,
                score_metric,
                answer_granularity,
                ipv6,
            ]
            batches.append(batch)

        logger.info(f"Running score calculation on:: {len(batches)} batches")

        batched_scores = pool.map(get_hostname_score, batches)

        for batch_score_per_subnet in batched_scores:
            for subnet, scores in batch_score_per_subnet.items():
                score_per_target_subnet[subnet] = scores

    dump_pickle(score_per_target_subnet, output_path)

    return score_per_target_subnet


def select_vp_per_subnet(
    target_subnet: str,
    target_score: dict,
    vps_per_subnet: dict,
    last_mile_delay_vp: dict,
) -> list:
    """
    for each VP /24 prefix, select one vp based on last mile delay measurements
    """
    # retrieve all vps belonging to subnets with highest mapping scores
    vp_per_subnet = []
    # target_score = sorted(target_score, key=lambda x: x[1], reverse=True)
    for subnet, score in target_score:
        # for fairness, do not take vps that are in the same subnet as the target
        if subnet == target_subnet:
            continue

        vps_in_subnet = vps_per_subnet[subnet]

        vps_delay_subnet = []
        for vp in vps_in_subnet:
            try:
                vps_delay_subnet.append((vp["addr"], last_mile_delay_vp[vp["addr"]]))
            except KeyError:
                continue

        # for each subnet, elect the VP with the lowest last mile delay
        if vps_delay_subnet:
            elected_subnet_vp_addr, _ = min(vps_delay_subnet, key=lambda x: x[-1])
            vp_per_subnet.append((elected_subnet_vp_addr, score))

    return vp_per_subnet


def filter_vps_last_mile_delay(
    ecs_vps: list[tuple],
    last_mile_delay: dict,
    rtt_threshold: int = 4,
) -> list[tuple]:
    """remove vps that have a high last mile delay"""
    filtered_vps = []
    for vp_addr, score in ecs_vps:
        try:
            min_rtt = last_mile_delay[vp_addr]

            if min_rtt < rtt_threshold:
                filtered_vps.append((vp_addr, score))
        except KeyError:
            continue

    return filtered_vps


def select_one_vp_per_as_city(
    raw_vp_selection: list,
    vp_coordinates: dict,
    last_mile_delay: dict,
    threshold: int = 40,
) -> list:
    """from a list of VP, select one per AS and per city"""
    filtered_vp_selection = []
    vps_per_as = defaultdict(list)
    for vp_addr, score in raw_vp_selection:
        _, _, vp_asn = vp_coordinates[vp_addr]
        try:
            last_mile_delay_vp = last_mile_delay[vp_addr]
        except KeyError:
            continue

        vps_per_as[vp_asn].append((vp_addr, last_mile_delay_vp, score))

    # select one VP per AS, take maximum VP score in AS
    selected_vps_per_as = defaultdict(list)
    for asn, vps in vps_per_as.items():
        vps_per_as[asn] = sorted(vps, key=lambda x: x[-1], reverse=True)
        for vp_i, last_mile_delay, score in vps_per_as[asn]:
            vp_i_lat, vp_i_lon, _ = vp_coordinates[vp_i]

            if not selected_vps_per_as[asn]:
                selected_vps_per_as[asn].append((vp_i, score))
                filtered_vp_selection.append((vp_i, score))
            else:
                already_found = False
                for vp_j, score in selected_vps_per_as[asn]:

                    vp_j_lat, vp_j_lon, _ = vp_coordinates[vp_j]

                    d = distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

                    if d < threshold:
                        already_found = True
                        break

                if not already_found:
                    selected_vps_per_as[asn].append((vp_i, score))
                    filtered_vp_selection.append((vp_i, score))

    return filtered_vp_selection


def get_vp_selection_per_target(
    output_path: Path, scores: Path, targets: list, vps: list[dict], ipv6: bool = False
) -> tuple[dict, dict]:

    if output_path.exists():
        vp_selection_per_target = load_pickle(output_path)
        return vp_selection_per_target

    if not ipv6:
        last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    else:
        # no filtering available in IPv6
        last_mile_delay = {vp["addr"]: 1 for vp in vps}

    vps_per_addr = {}
    vps_coordinates = {}
    vps_per_subnet = defaultdict(list)
    for vp in vps:
        vps_per_addr[vp["addr"]] = vp
        vps_per_subnet[vp["subnet"]].append(vp)
        vps_coordinates[vp["addr"]] = (vp["lat"], vp["lon"], vp["asn"])

    vp_selection_per_target = {}
    for target_addr in tqdm(targets):

        target_subnet = get_prefix_from_ip(target_addr, ipv6=ipv6)

        # retrieve vps score
        try:
            target_score: dict = scores[target_subnet]
        except KeyError:
            logger.error(f"cannot find target score for subnet : {target_subnet}")
            continue

        # get vps, function of their subnet ecs score
        vp_per_subnet = select_vp_per_subnet(
            target_subnet, target_score, vps_per_subnet, last_mile_delay
        )

        vp_selection_delay = filter_vps_last_mile_delay(
            vp_per_subnet, last_mile_delay, 2
        )

        vp_selection_per_target[target_addr] = select_one_vp_per_as_city(
            vp_selection_delay, vps_coordinates, last_mile_delay
        )

    dump_pickle(vp_selection_per_target, output_path)
    return vp_selection_per_target
