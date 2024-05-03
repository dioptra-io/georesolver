from pyasn import pyasn
from tqdm import tqdm
from loguru import logger

from geogiant.common.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
    load_targets,
    load_vps,
)
from geogiant.common.utils import (
    select_one_vp_per_as_city,
    filter_vps_last_mile_delay,
    get_parsed_vps,
    shortest_ping,
    get_vp_info,
    parse_target,
    EvalResults,
    TargetScores,
)
from geogiant.common.files_utils import load_json, load_pickle, dump_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_ecs_vps(
    target_subnet: str,
    target_score: dict,
    vps_per_subnet: dict,
    last_mile_delay_vp: dict,
    probing_budget: int = 50,
) -> list:
    """
    get the target score and extract best VPs function of the probing budget
    return 1 VP per subnet, TODO: get best connected VP per subnet
    """
    # retrieve all vps belonging to subnets with highest mapping scores
    ecs_vps = []
    for subnet, score in target_score:
        # for fairness, do not take vps that are in the same subnet as the target
        if subnet == target_subnet:
            continue

        vps_in_subnet = vps_per_subnet[subnet]

        vps_delay_subnet = []
        for vp in vps_in_subnet:
            try:
                vps_delay_subnet.append((vp, last_mile_delay_vp[vp]))
            except KeyError:
                continue

        # for each subnet, elect the VP with the lowest last mile delay
        if vps_delay_subnet:
            elected_subnet_vp_addr, _ = min(vps_delay_subnet, key=lambda x: x[-1])
            ecs_vps.append((elected_subnet_vp_addr, score))

        # take only a number of subnets up to probing budget
        if len(ecs_vps) >= probing_budget:
            break

    return ecs_vps


def get_no_ping_vp(
    target, target_score: list, vps_per_subnet: dict, vps_coordinates: dict
) -> dict:
    """return VP with maximum score"""
    target_subnet = get_prefix_from_ip(target["addr"])

    subnet, _ = target_score[0]

    for subnet, _ in target_score:
        if subnet == target_subnet:
            continue
        else:
            vp_addr = vps_per_subnet[subnet][0]
            break

    return get_vp_info(target, target_score, vp_addr, vps_coordinates)


def ecs_dns_vp_selection_eval(
    targets: list,
    vps_per_subnet: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    last_mile_delay: dict,
    vps_coordinates: dict,
    probing_budgets: list,
) -> tuple[dict, dict]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    for target in tqdm(targets):

        target = parse_target(target, asndb)

        try:
            target_scores: dict = subnet_scores[target["subnet"]]
        except KeyError:
            logger.error(f"cannot find target score for subnet : {target['subnet']}")
            continue

        result_per_metric = {}
        for metric, target_score in target_scores.items():

            # get vps, function of their subnet ecs score
            ecs_vps = get_ecs_vps(
                target["subnet"], target_score, vps_per_subnet, last_mile_delay, 5_00
            )

            # remove vps that have a high last mile delay
            ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

            # take only one address per city and per AS
            # TODO: select function of the last mile delay
            ecs_vps_per_budget = {}
            for budget in probing_budgets:
                ecs_vps_per_budget[budget] = select_one_vp_per_as_city(
                    ecs_vps, vps_coordinates, last_mile_delay
                )[:budget]

            # NOT PING GEOLOC
            no_ping_vp = get_no_ping_vp(
                target,
                target_score,
                vps_per_subnet,
                vps_coordinates,
            )

            # SHORTEST PING GEOLOC
            try:
                ecs_shortest_ping_per_budget = {}
                for budget, ecs_vps in ecs_vps_per_budget.items():

                    ecs_shortest_ping_per_budget[budget] = shortest_ping(
                        [addr for addr, _ in ecs_vps],
                        ping_vps_to_target[target["addr"]],
                    )

            except KeyError:
                logger.debug(f"No ping available for target:: {target['addr']}")
                continue

            ecs_shortest_ping_vp_per_budget = {}
            for budget, (
                ecs_shortest_ping_addr,
                ecs_min_rtt,
            ) in ecs_shortest_ping_per_budget.items():
                if not ecs_shortest_ping_addr:
                    continue
                ecs_shortest_ping_vp = get_vp_info(
                    target,
                    target_score,
                    ecs_shortest_ping_addr,
                    vps_coordinates,
                    ecs_min_rtt,
                )
                ecs_shortest_ping_vp_per_budget[budget] = ecs_shortest_ping_vp

            result_per_metric[metric] = {
                "ecs_shortest_ping_vp_per_budget": ecs_shortest_ping_vp_per_budget,
                "no_ping_vp": no_ping_vp,
                "ecs_scores": target_score[:50],
                "ecs_vps": ecs_vps,
            }

        results[target["addr"]] = {
            "target": target,
            "result_per_metric": result_per_metric,
        }

    return results


def main() -> None:
    probing_budgets = [5, 10, 20, 30, 50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.TRACEROUTES_LAST_MILE_DELAY
    )
    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.PING_VPS_TO_TARGET, removed_vps
    )
    targets = load_targets(clickhouse_settings.VPS_FILTERED)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)

    logger.info("BGP prefix score geoloc evaluation")

    score_files = [
        # "scores__AMAZON-02_1_greedy_per_cdn.pickle",
        # "scores__AMAZON-02_5_greedy_per_cdn.pickle",
        # "scores__AMAZON-02_10_greedy_per_cdn.pickle",
        # "scores__AMAZON-02_100_greedy_per_cdn.pickle",
        # "scores__AMAZON-02_500_greedy_per_cdn.pickle",
        # "scores__AMAZON-02_1000_greedy_per_cdn.pickle",
        # "scores__GOOGLE_1_max_bgp_prefixes.pickle",
        # "scores__GOOGLE_5_max_bgp_prefixes.pickle",
        # "scores__GOOGLE_10_max_bgp_prefixes.pickle",
        # "scores__GOOGLE_100_max_bgp_prefixes.pickle",
        # "scores__GOOGLE_500_max_bgp_prefixes.pickle",
        # "scores__GOOGLE_541_max_bgp_prefixes.pickle",
        "scores__all_cdns_10_hostname_per_cdn_max_bgp_prefix.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.2_bgp_prefix_threshold_2.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.2_bgp_prefix_threshold_5.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.6_bgp_prefix_threshold_2.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.2_bgp_prefix_threshold_5.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.2_bgp_prefix_threshold_10.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.6_bgp_prefix_threshold_2.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.6_bgp_prefix_threshold_10.pickle",
        # "scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_0.8_bgp_prefix_threshold_10.pickle",
    ]

    for score_file in score_files:

        scores: TargetScores = load_pickle(path_settings.RESULTS_PATH / score_file)

        logger.info(f"ECS evaluation for score:: {score_file}")

        results_answers = {}
        results_answer_subnets = {}
        results_answer_bgp_prefixes = {}
        if "answers" in scores.score_config["answer_granularities"]:
            results_answers = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        if "answer_subnets" in scores.score_config["answer_granularities"]:
            results_answer_subnets = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_subnets,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        if "answer_bgp_prefixes" in scores.score_config["answer_granularities"]:
            results_answer_bgp_prefixes = ecs_dns_vp_selection_eval(
                targets=targets,
                vps_per_subnet=vps_per_subnet,
                subnet_scores=scores.score_answer_bgp_prefixes,
                ping_vps_to_target=ping_vps_to_target,
                last_mile_delay=last_mile_delay,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
            )

        results = EvalResults(
            target_scores=scores,
            results_answers=results_answers,
            results_answer_subnets=results_answer_subnets,
            results_answer_bgp_prefixes=results_answer_bgp_prefixes,
        )

        output_file = (
            path_settings.RESULTS_PATH / f"{'results' + score_file.split('scores')[-1]}"
        )

        logger.info(f"output file:: {output_file}")

        dump_pickle(
            data=results,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()
