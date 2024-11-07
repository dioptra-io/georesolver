from pyasn import pyasn
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from collections import defaultdict

from georesolver.clickhouse.queries import (
    get_pings_per_target,
    get_min_rtt_per_vp,
)
from georesolver.agent.ping_process import get_ecs_vps
from georesolver.common.utils import (
    get_parsed_vps,
    shortest_ping,
    get_vp_info,
    parse_target,
    TargetScores,
)
from georesolver.common.geoloc import distance
from georesolver.common.files_utils import load_pickle
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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
        _, _, _, vp_asn = vp_coordinates[vp_addr]
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
            vp_i_lat, vp_i_lon, _, _ = vp_coordinates[vp_i]

            if not selected_vps_per_as[asn]:
                selected_vps_per_as[asn].append((vp_i, score))
                filtered_vp_selection.append((vp_i, score))
            else:
                already_found = False
                for vp_j, score in selected_vps_per_as[asn]:

                    vp_j_lat, vp_j_lon, _, _ = vp_coordinates[vp_j]

                    d = distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

                    if d < threshold:
                        already_found = True
                        break

                if not already_found:
                    selected_vps_per_as[asn].append((vp_i, score))
                    filtered_vp_selection.append((vp_i, score))

    return filtered_vp_selection


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


def get_no_ping_vp(
    target,
    target_score: list,
    vps_per_subnet: dict,
    vps_coordinates: dict,
    major_country: str = None,
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

    return get_vp_info(
        target,
        target_score,
        vp_addr,
        vps_coordinates,
        major_country=major_country,
    )


def ecs_dns_vp_selection_eval(
    targets: list,
    vps_per_subnet: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    last_mile_delay: dict,
    vps_coordinates: dict,
    probing_budgets: list,
    vps_country: dict[str] = None,
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
                target["subnet"], target_score, vps_per_subnet, last_mile_delay, 10_000
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

            country_no_ping = []
            if vps_country:
                countries = [
                    vps_country[vp_addr] for vp_addr, _ in ecs_vps_per_budget[50]
                ]
                major_country = max(set(countries), key=countries.count)
                proportion = countries.count(major_country) / len(countries)
                country_no_ping = (major_country, proportion)

            # NOT PING GEOLOC
            # no_ping_vp = get_no_ping_vp(
            #     target,
            #     target_score,
            #     vps_per_subnet,
            #     vps_coordinates,
            #     major_country=country_no_ping,
            # )

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
                # "no_ping_vp": no_ping_vp,
                "ecs_scores": ecs_vps_per_budget[budget],
                "ecs_vps": ecs_vps,
            }

        results[target["addr"]] = {
            "target": target,
            "result_per_metric": result_per_metric,
        }

    return results


def get_shortest_ping_geo_resolver(
    targets: list[str],
    vps: list[dict],
    score_file: Path,
    ping_table: str,
    removed_vps: list[str] = [],
    probing_budgets: list[int] = [50],
) -> dict:
    geo_resolver_sp = defaultdict(list)

    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    scores: TargetScores = load_pickle(score_file)
    scores = scores.score_answer_subnets

    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )

    ping_vps_to_target = get_pings_per_target(ping_table, removed_vps)

    for budget in probing_budgets:
        targets_shortest_ping = []
        for target in tqdm(targets):
            target_subnet = get_prefix_from_ip(target)
            target_scores = scores[target_subnet]

            # soi = detect_anycast(ping_vps_to_target[target], vp_distance_matrix)
            # if soi:
            #     logger.info(f"{target} is flagged as anycast")
            #     continue

            if not target_scores:
                logger.error(f"{target_subnet} does not have score")

            for _, target_score in target_scores.items():

                # get vps, function of their subnet ecs score
                ecs_vps = get_ecs_vps(
                    target_subnet,
                    target_score,
                    vps_per_subnet,
                    last_mile_delay,
                    5_00,
                )

                # remove vps that have a high last mile delay
                ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

                # take only one address per city and per AS
                ecs_vps_per_budget = []
                ecs_vps_per_budget = select_one_vp_per_as_city(
                    ecs_vps, vps_coordinates, last_mile_delay
                )[:budget]

                # SHORTEST PING GEOLOC
                try:
                    vp_shortest_ping_addr, shortest_ping_rtt = shortest_ping(
                        [vp_addr for vp_addr, _ in ecs_vps_per_budget],
                        ping_vps_to_target[target],
                    )
                    if vp_shortest_ping_addr:
                        targets_shortest_ping.append(
                            (target, vp_shortest_ping_addr, shortest_ping_rtt)
                        )

                except KeyError as e:
                    logger.debug(f"No ping available for target:: {target}:: {e}")
                    continue

        geo_resolver_sp[budget] = targets_shortest_ping

    return geo_resolver_sp
