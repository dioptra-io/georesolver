import asyncio
import numpy as np

from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import GetTargets, GetVPs, Query
from geogiant.analysis.plot import plot_median_error_per_finger_printing_method
from geogiant.common.geoloc import (
    distance,
    polygon_centroid,
    weighted_centroid,
)
from geogiant.common.files_utils import load_pickle, dump_pickle, load_csv
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


class OverallScore(Query):
    def statement(
        self,
        table_name: str,
        **kwargs,
    ) -> str:
        if hostname_filter := kwargs.get("hostname_filter"):
            hostname_filter = "".join([f",'{h}'" for h in hostname_filter])[1:]
            hostname_filter = f"AND hostname IN ({hostname_filter})"
        else:
            hostname_filter = ""
        if target_filter := kwargs.get("target_filter"):
            target_filter = "".join([f",toIPv4('{t}')" for t in target_filter])[1:]
        else:
            target_filter = ""
        if column_name := kwargs.get("column_name"):
            column_name = column_name
        else:
            raise RuntimeError(f"Column name parameter missing for {__class__}")

        return f"""
        WITH groupArray((toString(subnet_2), score)) AS subnet_scores
        SELECT
            toString(subnet_1) as target_subnet,
            arraySort(x -> -(x.2), subnet_scores) as scores
        FROM
        (
            SELECT
                t1.subnet AS subnet_1,
                t2.subnet AS subnet_2,
                length(arrayIntersect(t1.mapping, t2.mapping)) / least(length(t1.mapping), length(t2.mapping)) AS score
            FROM
            (
                SELECT
                    subnet,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE 
                    subnet IN ({target_filter})
                    {hostname_filter}
                GROUP BY subnet
            ) AS t1
            CROSS JOIN
            (
                SELECT
                    subnet,
                    groupUniqArray({column_name}) AS mapping
                FROM {self.settings.DATABASE}.{table_name}
                WHERE 
                    subnet NOT IN ({target_filter})
                    {hostname_filter}
                GROUP BY subnet
            ) AS t2
            WHERE t1.subnet != t2.subnet
        )
        WHERE subnet_1 IN ({target_filter})
        GROUP BY subnet_1
        """


async def get_score(
    target_subnet: list,
    column_name: str,
    hostname_filter: tuple[str],
) -> None:
    subnet_score = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await OverallScore().execute(
            client=client,
            table_name=clickhouse_settings.DNS_MAPPING_VPS_RAW,
            target_filter=target_subnet,
            column_name=column_name,
            hostname_filter=hostname_filter,
        )

    for row in resp:
        subnet_score[row["target_subnet"]] = row["scores"]

    return subnet_score


async def compute_scores(targets: list) -> None:

    hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames.csv")
    hostname_filter = [row.split(",")[0] for row in hostname_filter]

    logger.info(f"No ping geolocation using {len(hostname_filter)} hostnames")

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: BGP PREFIXES              #")
    logger.info("#############################################")
    bgp_prefix_score = await get_score(
        [target["target_subnet"] for target in targets],
        "answer_bgp_prefix",
        hostname_filter,
    )
    dump_pickle(
        data=bgp_prefix_score,
        output_file=path_settings.RESULTS_PATH / "bgp_prefix_score_1M_hostnames.pickle",
    )


def ecs_target_geoloc(
    vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> tuple[float, float]:
    """
    from a list of selected vps selected with ecs-dns,
    return an estimation of the target ip address
    """
    points = []
    for vp_addr in vps_assigned:
        try:
            vp_lat, vp_lon = vps_coordinate[vp_addr]
            points.append((vp_lat, vp_lon))
        except KeyError:
            logger.info(f"{vp_addr} missing from ripe atlas set")

    target_lat, target_lon = polygon_centroid(points)

    return (target_lat, target_lon)


def get_vp_weight(
    vp_i: str, vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> float:
    """
    get vp weight for centroid calculation.
    we take w = 1 / sum(d_i_j)
    """
    sum_d = 0
    vp_i_lat, vp_i_lon = vps_coordinate[vp_i]
    for vp_j in vps_assigned:
        if vp_j == vp_i:
            continue
        try:
            vp_j_lat, vp_j_lon = vps_coordinate[vp_j]
        except KeyError:
            continue

        sum_d += distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

    if not sum_d:
        return 0

    return 1 / sum_d


def weighted_ecs_target_geoloc(
    vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> tuple[float, float]:
    """
    from a list of selected vps selected with ecs-dns,
    return an estimation of the target ip address
    """
    points = []
    for vp_addr in vps_assigned:
        try:
            vp_lat, vp_lon = vps_coordinate[vp_addr]
            vp_weight = get_vp_weight(vp_addr, vps_assigned, vps_coordinate)
            points.append((vp_lat, vp_lon, vp_weight))

        except KeyError:
            logger.info(f"{vp_addr} missing from ripe atlas set")

    return weighted_centroid(points)


def best_ecs_target_geoloc(
    vps_assigned: list[str], vps_coordinate: dict[tuple]
) -> tuple[float, float]:
    """simply take the VP with the highest score as target geoloc proxy"""
    for vp_addr in vps_assigned:
        try:
            vp_lat, vp_lon = vps_coordinate[vp_addr]
            break

        except KeyError:
            logger.info(f"{vp_addr} missing from ripe atlas set")

    return (vp_lat, vp_lon)


def no_pings_eval(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    vps_coordinates: dict,
    probing_budget: int = 20,
) -> tuple[dict, dict]:
    results = {}
    w_results = {}
    b_results = {}
    for target in targets:
        target_addr = target["target_addr"]
        target_subnet = target["target_subnet"]
        target_lon, target_lat = target["lon"], target["lat"]
        target_scores = subnet_scores[target_subnet]

        # retrieve all vps belonging to subnets with highest mapping scores
        vps_assigned = []
        for vp_subnet, _ in target_scores[:probing_budget]:
            vps_assigned.extend([vp_addr for vp_addr in vps_subnet[vp_subnet]])

        # estimate target geoloc based on ecs score
        ecs_lat, ecs_lon = ecs_target_geoloc(vps_assigned, vps_coordinates)
        w_ecs_lat, w_ecs_lon = weighted_ecs_target_geoloc(vps_assigned, vps_coordinates)
        b_ecs_lat, b_ecs_lon = best_ecs_target_geoloc(vps_assigned, vps_coordinates)

        # compare true target geolocation with estimated one
        d_error = distance(target_lat, ecs_lat, target_lon, ecs_lon)
        b_d_error = distance(target_lat, b_ecs_lat, target_lon, b_ecs_lon)

        results[target_addr] = d_error
        b_results[target_addr] = b_d_error

        # TODO: find why sometimes no output for weighted geoloc
        if not w_ecs_lat or not w_ecs_lon:
            continue

        w_d_error = distance(target_lat, w_ecs_lat, target_lon, w_ecs_lon)
        w_results[target_addr] = w_d_error

    return results, w_results, b_results


def get_metrics(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    vps_coordinates: dict,
    probing_budgets: list,
) -> dict:
    """get geolocation error function of the probing budget"""
    overall_results = {}
    for budget in probing_budgets:
        if budget == 0:
            continue

        no_ping_r, w_no_ping_r, b_no_ping_r = no_pings_eval(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores,
            vps_coordinates=vps_coordinates,
            probing_budget=budget,
        )

        median_d = round(np.median([d_error for d_error in no_ping_r.values()]), 2)
        w_median_d = round(np.median([d_error for d_error in w_no_ping_r.values()]), 2)
        b_median_d = round(np.median([d_error for d_error in b_no_ping_r.values()]), 2)

        overall_results[budget] = (median_d, w_median_d)

        # debugging
        if budget in [5, 10, 20, 30, 50, 100]:
            logger.info(f"probing budget: {budget}")
            logger.info(f"Median error: {median_d}")
            logger.info(f"Weight Median error: {w_median_d}")
            logger.info(f"Highest score vp Median error: {b_median_d}")

    return overall_results


async def main() -> None:
    probing_budgets = [1]
    probing_budgets.extend([i for i in range(5, 50, 1)])
    probing_budgets.extend([i for i in range(50, 100, 10)])

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        targets = await GetTargets().execute(
            client=client,
            table_name=clickhouse_settings.VPS_RAW,
        )

        vps = await GetVPs().execute(
            client=client,
            table_name=clickhouse_settings.VPS_RAW,
        )

    # parse
    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    for row in vps:
        addr = row["vp_addr"]
        subnet = row["vp_subnet"]
        vp_lon, vp_lat = row["lon"], row["lat"]

        vps_coordinates[addr] = (vp_lat, vp_lon)
        vps_subnet[subnet].append(addr)

    if not (
        path_settings.RESULTS_PATH / "bgp_prefix_score_1M_hostnames.pickle"
    ).exists():
        logger.info("BGP prefix score with all hostnames not yet calculated")
        await compute_scores(targets)

    subnet_scores = load_pickle(
        path_settings.RESULTS_PATH / "bgp_prefix_score_1M_hostnames.pickle"
    )

    eval_results = {}
    logger.info("BGP prefix score geoloc evaluation")
    eval_results["bgp_prefix"] = get_metrics(
        targets=targets,
        vps_subnet=vps_subnet,
        subnet_scores=subnet_scores,
        vps_coordinates=vps_coordinates,
        probing_budgets=probing_budgets,
    )

    plot_median_error_per_finger_printing_method(
        eval_results, out_file="no_pings_evaluation.pdf"
    )


if __name__ == "__main__":
    asyncio.run(main())
