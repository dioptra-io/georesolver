import asyncio
import numpy as np

from pyasn import pyasn
from loguru import logger
from collections import defaultdict

from geogiant.ecs_vp_selection.query import (
    get_pop_info,
    get_subnets_mapping,
    get_subnet_per_pop,
    get_pop_per_hostname,
    get_pings_per_target,
)
from geogiant.ecs_vp_selection.utils import (
    ResultsScore,
    get_ecs_pings,
    get_parsed_vps,
    get_vp_to_pops_dst,
)
from geogiant.ecs_vp_selection.utils import select_one_vp_per_as_city
from geogiant.common.geoloc import distance
from geogiant.common.settings import PathSettings, ClickhouseSettings
from geogiant.common.files_utils import (
    load_json,
    load_pickle,
    load_csv,
    dump_pickle,
    dump_json,
)
from geogiant.common.ip_addresses_utils import (
    get_prefix_from_ip,
    route_view_bgp_prefix,
    get_addr_granularity,
)

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def check_ecs_misallocation(
    addr: str,
    subnet: str,
    addr_lat: float,
    addr_lon: float,
    subnet_mapping: dict[str, list],
    pop_per_hostname: dict,
    subnet_per_pop: dict,
) -> None:
    # find the best PoP for each hostname
    hostname_misallocation = []
    total_hostnames = len(subnet_mapping)
    if len(subnet_mapping) == 0:
        logger.error(f"Cannot find target mapping: {subnet}")
        return 0

    for hostname, mapping in subnet_mapping.items():
        hostname_pops: dict = pop_per_hostname[hostname]

        # get best pop for vp
        addr_to_all_pop = []
        for pop_subnet, (pop_lat, pop_lon) in hostname_pops.items():
            if pop_lat != -1 and pop_lon != -1:
                d = distance(addr_lat, pop_lat, addr_lon, pop_lon)
                addr_to_all_pop.append((pop_subnet, d, pop_lat, pop_lon))

        # get elected best pop
        addr_to_ecs_pop = []
        for pop_subnet in mapping:
            pop_lat, pop_lon = hostname_pops[pop_subnet]
            if pop_lat != -1 and pop_lon != -1:
                d = distance(addr_lat, pop_lat, addr_lon, pop_lon)
                addr_to_ecs_pop.append((pop_subnet, d, pop_lat, pop_lon))

        # get min
        if addr_to_all_pop and addr_to_ecs_pop:
            closest_pop = min(addr_to_all_pop, key=lambda x: x[1])
            ecs_closest_pop = min(addr_to_ecs_pop, key=lambda x: x[1])

            closest_pop_lat = closest_pop[2]
            closest_pop_lon = closest_pop[3]

            closest_pop_subnets = subnet_per_pop[hostname][
                (closest_pop_lat, closest_pop_lon)
            ]

            if abs(closest_pop[1] - ecs_closest_pop[1]) > 1000:
                # compare the two
                # logger.info("#############################################")
                # logger.info(f"{addr=}")
                # logger.info(f"{subnet=}")
                # logger.info(f"{hostname=}")
                # logger.info(f"{closest_pop=}")
                # logger.info(f"{ecs_closest_pop=}")
                # logger.info(f"{closest_pop_subnets=}")
                # logger.info(f"{subnet_mapping[subnet][hostname]=}")
                # logger.info("#############################################")
                hostname_misallocation.append(hostname)

    # return the percentage of misallocated hostnames
    return len(hostname_misallocation) * 100 / total_hostnames


def ecs_dns_vp_selection_eval(
    targets: list,
    parsed_vps: dict,
    granularity: str,
    scores: dict,
    inconsistent_mapping: list,
    ping_vps_to_target: dict,
    vp_coordinates: dict,
    subnet_mapping: dict[str, list],
    pop_per_hostname: dict,
    subnet_per_pop: dict,
    probing_budget: int,
) -> tuple[list, set]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    unmapped_t = set()
    for target in targets:

        target_addr = target["address_v4"]
        target_subnet = get_prefix_from_ip(target_addr)
        target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)
        target_granularity = get_addr_granularity(target_addr, granularity, asndb)
        target_lon, target_lat = target["geometry"]["coordinates"]

        if target_subnet in inconsistent_mapping:
            pass

        # get target/vps dns mapping score
        try:
            target_scores = scores[target_granularity]
        except KeyError:
            unmapped_t.add(target_addr)
            continue

        # check target resolution
        target_misallocation = check_ecs_misallocation(
            addr=target_addr,
            subnet=target_subnet,
            addr_lat=target_lat,
            addr_lon=target_lon,
            subnet_mapping=subnet_mapping[target_subnet],
            pop_per_hostname=pop_per_hostname,
            subnet_per_pop=subnet_per_pop,
        )

        # retrieve all vps belonging to subnets with highest mapping scores
        vps_assigned = []
        for vp_granularity, _ in target_scores[:probing_budget]:
            # TODO: take VP within the same /24 based on connectivity
            vps_assigned.append(parsed_vps[vp_granularity][0])

        vps_assigned = select_one_vp_per_as_city(vps_assigned, vp_coordinates)

        # get vps and pings associated to the dns mapping selection
        try:
            vp_selection = get_ecs_pings(
                target_associated_vps=vps_assigned,
                ping_to_target=ping_vps_to_target[target_addr],
            )
        except KeyError:
            unmapped_t.add(target_addr)
            continue

        # if no vps, save target as unmapped
        if not vp_selection:
            unmapped_t.add(target_addr)
            continue

        # BEST ELECTED VP
        elected_vp, elected_rtt = min(vp_selection, key=lambda x: x[-1])
        elected_subnet = get_prefix_from_ip(elected_vp)
        elect_vp_lat, elect_vp_lon, _ = vp_coordinates[elected_vp]
        d_elected_error = distance(target_lat, elect_vp_lat, target_lon, elect_vp_lon)
        elected_vp_score = -1
        elected_vp_index = -1
        for i, (vp_granularity, score) in enumerate(target_scores):
            if get_addr_granularity(elected_vp, granularity, asndb) == vp_granularity:
                elected_vp_score = score
                elected_vp_index = i
                break

        # BEST VP
        best_vp_addr, best_rtt = min(
            ping_vps_to_target[target_addr], key=lambda x: x[-1]
        )
        best_vp_subnet = get_prefix_from_ip(best_vp_addr)
        try:
            best_vp_lat, best_vp_lon, _ = vp_coordinates[best_vp_addr]
        except KeyError:
            continue
        best_d_error = distance(target_lat, best_vp_lat, target_lon, best_vp_lon)
        best_vp_score = -1
        best_vp_index = -1
        for i, (vp_subnet, score) in enumerate(target_scores):
            if best_vp_subnet == vp_subnet:
                best_vp_score = score
                best_vp_index = i
                break

        # MAX SCORE VP
        max_score_subnet, max_score = target_scores[0]
        max_score_vp = vps_assigned[0]
        max_score_rtt = max_score_vp[-1]
        max_score_vp_lat, max_score_vp_lon, _ = vp_coordinates[max_score_vp]
        d_max_score_error = distance(
            target_lat, max_score_vp_lat, target_lon, max_score_vp_lon
        )

        results[target_addr] = {
            "target_subnet": target_subnet,
            "target_bgp_prefix": target_bgp_prefix,
            "target_misallocation": target_misallocation,
            "elected_vp": elected_vp,
            "elected_subnet": elected_subnet,
            "elected_vp_score": elected_vp_score,
            "elected_vp_index": elected_vp_index,
            "elected_rtt": elected_rtt,
            "elected_d_error": d_elected_error,
            "best_vp": best_vp_addr,
            "best_subnet": best_vp_subnet,
            "best_vp_score": best_vp_score,
            "best_vp_index": best_vp_index,
            "best_min_rtt": best_rtt,
            "best_d_error": best_d_error,
            "max_score_vp": max_score_vp,
            "max_score_subnet": max_score_subnet,
            "max_score": max_score,
            "max_score_rtt": max_score_rtt,
            "max_score_d_error": d_max_score_error,
            "first_elected_vps_scores": target_scores[:50],
            "vp_selection": vps_assigned,
        }

    return results, unmapped_t


def get_valid_hostnames_filter() -> str:
    """get clickhouse query subset for hostname filtering"""
    hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames_cdn.csv")
    hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
    return f"AND hostname IN ({hostname_filter})"


def get_metrics_vp_selection(
    targets: list,
    parsed_vps: dict,
    granularity: str,
    ping_vps_to_target: dict,
    scores: ResultsScore,
    vp_coordinates: dict,
    probing_budgets: list,
    subnet_mapping: dict,
    pop_per_hostname: dict,
    subnet_per_pop: dict,
    verbose: bool = False,
) -> dict:
    """calculate some basic metrics from a vp selection"""
    overall_results = {}
    wrongful_geoloc = {}
    for budget in probing_budgets:
        if budget == 0:
            continue

        subnet_with_inconsistent_mapping = [
            get_prefix_from_ip(t["address_v4"]) for t in scores.inconsistent_mappings
        ]
        overall_results[budget], unmapped_t = ecs_dns_vp_selection_eval(
            targets=targets,
            parsed_vps=parsed_vps,
            granularity=granularity,
            scores=scores.scores,
            inconsistent_mapping=subnet_with_inconsistent_mapping,
            ping_vps_to_target=ping_vps_to_target,
            vp_coordinates=vp_coordinates,
            subnet_mapping=subnet_mapping,
            pop_per_hostname=pop_per_hostname,
            subnet_per_pop=subnet_per_pop,
            probing_budget=budget,
        )

        median_distance = round(
            np.median(
                [
                    r["elected_d_error"]
                    for r in overall_results[budget].values()
                    if r["target_subnet"] not in subnet_with_inconsistent_mapping
                ]
            ),
            2,
        )
        logger.info(f"probing budget={budget} Median dist={median_distance}")

        wrongful_geoloc[budget] = {}
        for target_addr, target_results in overall_results[budget].items():
            diff_error = (
                target_results["elected_d_error"] - target_results["best_d_error"]
            )
            if diff_error > 100 and target_results["best_d_error"] < 100:
                wrongful_geoloc[budget][target_addr] = target_results

    return overall_results, wrongful_geoloc


async def main(targets: list, vps: list) -> None:
    answer = False
    subnet = True
    bgp = False
    pop = False
    cdn = False

    extend_vp_mapping = False
    filtered_hostname = True
    verbose = False

    asndb = pyasn(str(path_settings.RIB_TABLE))
    probing_budgets = [1, 10, 50, 100, 200, 500]

    hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames_cdn.csv")
    vps_subnet, vps_bgp_prefix, vps_coordinates = get_parsed_vps(vps, asndb)

    subnet_per_pop = await get_subnet_per_pop(
        dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
        hostname_filter=hostname_filter,
    )
    pop_per_hostname = await get_pop_per_hostname(
        dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
        hostname_filter=hostname_filter,
    )
    ping_vps_to_target = await get_pings_per_target(
        clickhouse_settings.OLD_PING_VPS_TO_TARGET
    )
    pop_info = await get_pop_info(
        answer_granularity="answer_subnet",
        dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
        hostname_filter=hostname_filter,
    )

    dump_json(pop_info, path_settings.DATASET / "pop_info.json")

    targets_mapping = await get_subnets_mapping(
        answer_granularity="answer_subnet",
        dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
        hostname_filter=hostname_filter,
        subnets=[get_prefix_from_ip(t["address_v4"]) for t in targets],
    )

    base_input_file = f"score_{'extended_mapping' if extend_vp_mapping else 'not_extended'}_{'filtered_hostname' if filtered_hostname else 'no_filtered'}"
    base_output_file = f"results_{'extended_mapping' if extend_vp_mapping else 'not_extended'}_{'filtered_hostname' if filtered_hostname else 'no_filtered'}"

    if answer:
        answer_results = {}
        answer_score: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH / (base_input_file + "_answer.pickle")
        )
        logger.info(f"Score file: {(base_input_file + '_answer.pickle')}")
        logger.info("Answer score evaluation")
        answer_results["results"], answer_results["wrongful_geoloc"] = (
            get_metrics_vp_selection(
                targets=targets,
                parsed_vps=vps_subnet,
                granularity="answer",
                ping_vps_to_target=ping_vps_to_target,
                scores=answer_score,
                vp_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
                subnet_mapping=targets_mapping,
                pop_per_hostname=pop_per_hostname,
                subnet_per_pop=subnet_per_pop,
                verbose=verbose,
            )
        )

        dump_pickle(
            answer_results,
            path_settings.RESULTS_PATH / (base_output_file + "_answer.pickle"),
        )

    if subnet:
        subnet_results = {}
        subnet_scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH / (base_input_file + "_subnet.pickle")
        )
        logger.info(f"Score file: {(base_input_file + '_subnet.pickle')}")
        logger.info("Subnet score evaluation")
        subnet_results["results"], subnet_results["wrongful_geoloc"] = (
            get_metrics_vp_selection(
                targets=targets,
                parsed_vps=vps_subnet,
                granularity="client_subnet",
                ping_vps_to_target=ping_vps_to_target,
                scores=subnet_scores,
                vp_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
                subnet_mapping=targets_mapping,
                pop_per_hostname=pop_per_hostname,
                subnet_per_pop=subnet_per_pop,
                verbose=verbose,
            )
        )

        dump_pickle(
            subnet_results,
            path_settings.RESULTS_PATH / (base_output_file + "_subnet.pickle"),
        )

    if bgp:
        bgp_results = {}
        logger.info("BGP score evaluation")
        bgp_scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH / (base_input_file + "_bgp.pickle")
        )
        bgp_results["results"], bgp_results["wrongful_geoloc"] = (
            get_metrics_vp_selection(
                targets=targets,
                parsed_vps=vps_subnet,
                granularity="client_subnet",
                ping_vps_to_target=ping_vps_to_target,
                scores=bgp_scores,
                vp_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
                verbose=verbose,
            )
        )

        dump_pickle(
            bgp_results,
            path_settings.RESULTS_PATH / (base_output_file + "_bgp.pickle"),
        )

    if pop:
        pop_city_results = {}
        pop_city_scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH / "score_True_True_pop_city.pickle"
        )
        logger.info("Subnet score evaluation")
        pop_city_results["results"], pop_city_results["wrongful_geoloc"] = (
            get_metrics_vp_selection(
                targets=targets,
                parsed_vps=vps_subnet,
                granularity="pop_city",
                ping_vps_to_target=ping_vps_to_target,
                scores=pop_city_scores,
                vp_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
                verbose=verbose,
            )
        )

        dump_pickle(
            pop_city_results,
            path_settings.RESULTS_PATH / "pop_city_results.pickle",
        )

    if cdn:
        cdns = ["facebook", "google", "amazon", "cdn_networks", "apple"]

        results_per_cdn = {}
        wrongful_geoloc_per_cdn = {}
        for cdn in cdns:
            logger.info(f"Compute scores for {cdn}")
            cdn_scores = load_pickle(path_settings.RESULTS_PATH / f"score_{cdn}.pickle")

            results_per_cdn[cdn], wrongful_geoloc_per_cdn[cdn] = (
                get_metrics_vp_selection(
                    targets=targets,
                    parsed_vps=vps_subnet,
                    granularity="client_subnet",
                    ping_vps_to_target=ping_vps_to_target,
                    scores=cdn_scores["client_subnet"]["answer_subnet"],
                    vp_coordinates=vps_coordinates,
                    probing_budgets=probing_budgets,
                    verbose=verbose,
                )
            )

            dump_pickle(
                data=results_per_cdn,
                output_file=path_settings.RESULTS_PATH / f"results_per_cdn.pickle",
            )

            dump_pickle(
                wrongful_geoloc_per_cdn,
                path_settings.RESULTS_PATH / "wrongful_geoloc_per_cdn.pickle",
            )


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(main(targets, vps))
