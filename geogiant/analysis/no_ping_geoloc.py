import asyncio
import numpy as np
import matplotlib.path as mpltPath

from pyasn import pyasn
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

from geogiant.ecs_vp_selection.query import (
    get_pings_per_target,
    load_targets,
    load_vps,
)
from geogiant.ecs_vp_selection.utils import (
    get_ecs_pings,
    select_one_vp_per_as_city,
    get_parsed_vps,
    ResultsScore,
)
from geogiant.common.geoloc import distance
from geogiant.common.files_utils import load_pickle, dump_pickle, load_json
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_vp_in_cluster(
    clusters: dict,
    vp_selection: list,
    vps_coordinates: dict,
) -> dict:
    """return VPs present within the main cluster, with their score"""
    vp_in_cluster = defaultdict(list)
    for vp_addr, score in vp_selection:
        vp_lat, vp_lon, _ = vps_coordinates[vp_addr]

        for label, cluster_coordinates in clusters.items():
            if label == -1:
                continue
            for lat, lon in cluster_coordinates:
                if vp_lat == lat and vp_lon == lon:
                    vp_in_cluster[label].append((vp_addr, lat, lon, score))

    return vp_in_cluster


def get_cluster(vp_selection: list, vps_coordinates: dict) -> list:
    """based on ECS selected VPs, cluster VPs"""
    selected_vp_coordinates = []
    for vp_addr, _ in vp_selection:
        vp_lat, vp_lon, _ = vps_coordinates[vp_addr]

        selected_vp_coordinates.append([vp_lat, vp_lon])

    selected_vp_coordinates = np.array(selected_vp_coordinates, dtype=float)

    if selected_vp_coordinates.any():
        db = DBSCAN(eps=50 / 6371.0, min_samples=5, metric="haversine").fit(
            np.radians(selected_vp_coordinates)
        )
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        unique_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        clusters = {}
        for k in unique_labels:
            class_member_mask = labels == k
            clusters[k] = selected_vp_coordinates[class_member_mask & core_samples_mask]

        return clusters, n_clusters_, n_noise_


def get_all_vps_in_cluster(cluster_points: np.array, vps_coordinates: dict) -> list:
    """return all VPs present within a given cluster"""
    all_vps_in_cluster = set()
    cluster_polygon = Polygon(cluster_points)

    for vp_addr, (vp_lat, vp_lon, _) in vps_coordinates.items():
        point = Point((vp_lat, vp_lon))

        if point.within(cluster_polygon):
            all_vps_in_cluster.add((vp_addr, vp_lat, vp_lon))

    return all_vps_in_cluster


def find_exterior_points(points):
    # Create a numpy array from the list of points
    points_array = np.array(points)

    # Compute the convex hull
    hull = ConvexHull(points_array)

    # Extract the indices of exterior points
    exterior_indices = hull.vertices

    # Get the exterior points
    exterior_points = points_array[exterior_indices]

    return exterior_points, hull


def ecs_dns_vp_selection_eval(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    vps_coordinates: dict,
    probing_budget: int = 20,
) -> tuple[dict, dict]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    for target in tqdm(targets):
        target_addr = target["address_v4"]
        target_subnet = get_prefix_from_ip(target_addr)
        target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)
        target_lon, target_lat = target["lat"], target["lon"]

        try:
            target_scores = subnet_scores[target_subnet]
        except KeyError:
            logger.error(f"cannot find target score for subnet : {target_subnet}")

        # retrieve all vps belonging to subnets with highest mapping scores
        vps_assigned = []
        for vp_subnet, _ in target_scores[:probing_budget]:
            vps_in_subnet = vps_subnet[vp_subnet]
            if not vps_in_subnet:
                continue
            vps_assigned.append(vps_in_subnet[0])

        # vps_assigned = select_one_vp_per_as_city(vps_assigned, vps_coordinates)

        # CLUSTER SELECTION
        first_50_vps = []
        for vp_subnet, score in target_scores[:50]:
            vps_in_subnet = vps_subnet[vp_subnet]
            if not vps_in_subnet:
                continue
            first_50_vps.append((vps_in_subnet[0], score))

        clusters, n_clusters, n_noise = get_cluster(first_50_vps, vps_coordinates)
        vp_per_cluster = get_vp_in_cluster(clusters, first_50_vps, vps_coordinates)

        cluster_assigned = []
        cluster_scores = {}
        # for label, vp_in_cluster in vp_per_cluster.items():
        #     if label == -1:
        #         continue

        #     # avoid small clusters
        #     if len(vp_in_cluster) < 3:
        #         continue

        #     # get cluster scores
        #     cluster_poly, _ = find_exterior_points(
        #         [(vp[1], vp[2]) for vp in vp_in_cluster]
        #     )

        #     # get all vps in cluster poly
        #     all_vps_poly_cluster = get_all_vps_in_cluster(cluster_poly, vps_coordinates)

        #     # we cannot find vps in cluster polygon?
        #     if not all_vps_poly_cluster:
        #         continue

        #     # get the cluster score
        #     cluster_score = len(vp_in_cluster) / len(all_vps_poly_cluster)
        #     cluster_scores[label] = cluster_score

        #     cluster_assigned.extend(
        #         [vp[0] for i, vp in enumerate(vp_in_cluster) if i < 10]
        #     )

        # get vps and pings associated to the dns mapping selection
        try:
            vp_selection = get_ecs_pings(
                target_associated_vps=vps_assigned,
                ping_to_target=ping_vps_to_target[target_addr],
            )
        except KeyError:
            continue

        # get ping for cluster selection
        # try:
        #     cluster_vp_selection = get_ecs_pings(
        #         target_associated_vps=cluster_assigned,
        #         ping_to_target=ping_vps_to_target[target_addr],
        #     )
        # except KeyError:
        #     continue

        if not vp_selection:
            continue

        # if not cluster_vp_selection:
        #     continue

        # best VP cluster
        # cluster_elected_vp, cluster_elected_rtt = min(
        #     cluster_vp_selection, key=lambda x: x[-1]
        # )
        # cluster_elected_subnet = get_prefix_from_ip(cluster_elected_vp)
        # cluster_elect_vp_lat, cluster_elect_vp_lon, _ = vps_coordinates[
        #     cluster_elected_vp
        # ]
        # d_cluster_elected_error = distance(
        #     target_lat, cluster_elect_vp_lat, target_lon, cluster_elect_vp_lon
        # )
        # cluster_elected_vp_score = -1
        # cluster_elected_vp_index = -1
        # for i, (vp_subnet, score) in enumerate(target_scores):
        #     if cluster_elected_subnet == vp_subnet:
        #         cluster_elected_vp_score = score
        #         cluster_elected_vp_index = i
        #         break

        # MAX SCORE VP
        max_score_subnet, max_score = target_scores[0]
        max_score_vp = vps_subnet[max_score_subnet][0]
        max_score_rtt = max_score_vp[-1]  # TODO: find rtt, this is score
        max_score_vp_lat, max_score_vp_lon, _ = vps_coordinates[max_score_vp]
        d_max_score_error = distance(
            target_lat, max_score_vp_lat, target_lon, max_score_vp_lon
        )

        # TODO: find main cluster when multiple are returned
        # CLUSTER MAX SCORE VP
        # best_cluster_label, best_cluster_score = max(
        #     cluster_scores.items(), key=lambda x: x[1]
        # )
        # vp_in_cluster = vp_per_cluster[best_cluster_label]

        # take max score cluster
        if len(vp_per_cluster) == 1:
            for label, vp_in_cluster in vp_per_cluster.items():
                if label == -1:
                    continue
                (
                    cluster_max_score_vp,
                    cluster_max_score_vp_lat,
                    cluster_max_score_vp_lon,
                    cluster_max_score,
                ) = max(vp_in_cluster, key=lambda x: x[-1])

                cluster_max_score_d_error = distance(
                    target_lat,
                    cluster_max_score_vp_lat,
                    target_lon,
                    cluster_max_score_vp_lon,
                )

                cluster_max_score_subnet = get_prefix_from_ip(cluster_max_score_vp)
            else:
                cluster_max_score_vp = max_score_vp
                cluster_max_score_subnet = max_score_subnet
                cluster_max_score_vp_lat = max_score_vp_lat
                cluster_max_score_vp_lon = max_score_vp_lon
                cluster_max_score = max_score

        # ECS VP
        elected_vp, elected_rtt = min(vp_selection, key=lambda x: x[-1])
        elected_subnet = get_prefix_from_ip(elected_vp)
        elect_vp_lat, elect_vp_lon, _ = vps_coordinates[elected_vp]
        d_elected_error = distance(target_lat, elect_vp_lat, target_lon, elect_vp_lon)
        elected_vp_score = -1
        elected_vp_index = -1
        for i, (vp_subnet, score) in enumerate(target_scores):
            if elected_subnet == vp_subnet:
                elected_vp_score = score
                elected_vp_index = i
                break

        # REF VP
        all_vps_selection = sorted(ping_vps_to_target[target_addr], key=lambda x: x[-1])
        for ref_vp_addr, ref_min_rtt in all_vps_selection:
            try:
                ref_vp_lat, ref_vp_lon, _ = vps_coordinates[ref_vp_addr]
                break
            except KeyError:
                pass
        ref_vp_subnet = get_prefix_from_ip(ref_vp_addr)
        ref_d_error = distance(target_lat, ref_vp_lat, target_lon, ref_vp_lon)
        ref_vp_score = -1
        ref_vp_index = -1
        for i, (vp_subnet, score) in enumerate(target_scores):
            if ref_vp_subnet == vp_subnet:
                ref_vp_score = score
                ref_vp_index = i
                break

        results[target_addr] = {
            "target_subnet": target_subnet,
            "target_bgp_prefix": target_bgp_prefix,
            "elected_vp": elected_vp,
            "elected_vp_subnet": elected_subnet,
            "elected_vp_score": elected_vp_score,
            "elected_vp_index": elected_vp_index,
            "elected_rtt": elected_rtt,
            "elected_d_error": d_elected_error,
            "elected_vp": elected_vp,
            "cluster_elected_vp": cluster_elected_vp,
            "cluster_elected_vp_subnet": cluster_elected_subnet,
            "cluster_elected_vp_score": cluster_elected_vp_score,
            "cluster_elected_vp_index": cluster_elected_vp_index,
            "cluster_elected_rtt": cluster_elected_rtt,
            "cluster_elected_d_error": d_cluster_elected_error,
            "cluster_vps": cluster_assigned,
            "ref_vp": ref_vp_addr,
            "ref_vp_subnet": ref_vp_subnet,
            "ref_vp_score": ref_vp_score,
            "ref_vp_index": ref_vp_index,
            "ref_min_rtt": ref_min_rtt,
            "ref_d_error": ref_d_error,
            "max_score_vp": max_score_vp,
            "max_score_subnet": max_score_subnet,
            "max_score": max_score,
            "max_score_rtt": None,
            "cluster_max_score_vp": cluster_max_score_vp,
            "cluster_max_score_d_error": cluster_max_score_d_error,
            "cluster_max_score_subnet": cluster_max_score_subnet,
            "cluster_max_score": cluster_max_score,
            "cluster_max_score_rtt": None,
            "max_score_d_error": d_max_score_error,
            "first_elected_vps_scores": target_scores[:50],
            "vp_selection": vps_assigned,
        }

    return results


def get_metrics(
    targets: list,
    vps_subnet: dict,
    subnet_scores: dict,
    vps_coordinates: dict,
    probing_budgets: list,
    ping_vps_to_target: dict,
) -> dict:
    """get geolocation error function of the probing budget"""
    results = {}
    for budget in probing_budgets:

        results[budget] = ecs_dns_vp_selection_eval(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores,
            vps_coordinates=vps_coordinates,
            probing_budget=budget,
            ping_vps_to_target=ping_vps_to_target,
        )

    return results


async def main() -> None:
    greedy_cdn = False
    greedy_bgp = False
    max_bgp_per_cdn = True
    eval_per_cdn = False
    answer_granularity = "answer_bgp_prefix"
    probing_budgets = [1, 5, 10, 50]
    probing_budgets = [50]
    asndb = pyasn(str(path_settings.RIB_TABLE))

    ping_vps_to_target = await get_pings_per_target(
        clickhouse_settings.OLD_PING_VPS_TO_TARGET
    )
    targets = await load_targets(clickhouse_settings.VPS_RAW)
    vps = await load_vps(clickhouse_settings.VPS_RAW)

    vps_subnet, vps_bgp_prefix, vps_coordinates = get_parsed_vps(vps, asndb)

    eval_results = {}
    logger.info("BGP prefix score geoloc evaluation")

    if greedy_cdn:
        subnet_scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH
            / f"score_not_extended_filtered_hostname_1M_hostnames_greedy_cdn_answer_bgp_prefix.pickle"
        )

        eval_results[answer_granularity] = get_metrics(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores.scores,
            vps_coordinates=vps_coordinates,
            probing_budgets=probing_budgets,
            ping_vps_to_target=ping_vps_to_target,
        )

        dump_pickle(
            data=eval_results,
            output_file=path_settings.RESULTS_PATH
            / f"results_1M_hostnames_answer_bgp_prefix_greedy_cdn.pickle",
        )
    if greedy_bgp:
        subnet_scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH
            / f"score_not_extended_filtered_hostname_1M_hostnames_greedy_selection_bgp_answer_bgp_prefix.pickle"
        )

        eval_results[answer_granularity] = get_metrics(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores.scores,
            vps_coordinates=vps_coordinates,
            probing_budgets=probing_budgets,
            ping_vps_to_target=ping_vps_to_target,
        )

        dump_pickle(
            data=eval_results,
            output_file=path_settings.RESULTS_PATH
            / f"results_1M_hostnames_answer_bgp_prefix_greedy_bgp.pickle",
        )

    if max_bgp_per_cdn:
        subnet_scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH
            / f"score_not_extended_filtered_hostname_1M_hostnames_max_bgp_prefix_per_cdn_answer_bgp_prefix.pickle"
        )

        eval_results[answer_granularity] = get_metrics(
            targets=targets,
            vps_subnet=vps_subnet,
            subnet_scores=subnet_scores.scores,
            vps_coordinates=vps_coordinates,
            probing_budgets=probing_budgets,
            ping_vps_to_target=ping_vps_to_target,
        )

        dump_pickle(
            data=eval_results,
            output_file=path_settings.RESULTS_PATH
            / f"results_1M_hostnames_answer_bgp_prefix_max_bgp_prefix.pickle",
        )

    if eval_per_cdn:
        # per cdn analysis
        hostname_per_cdn = load_json(path_settings.DATASET / "hostname_per_cdn_1M.json")

        cdn_eval_results = {}
        for cdn, hostnames in hostname_per_cdn.items():
            logger.info(f"Evaluation for:: {cdn=}, {hostnames=}")

            cdn_score: ResultsScore = load_pickle(
                path_settings.RESULTS_PATH
                / f"score_not_extended_filtered_hostname_1M_hostnames_{answer_granularity}_{cdn}.pickle"
            )

            cdn_eval_results[cdn] = get_metrics(
                targets=targets,
                vps_subnet=vps_subnet,
                subnet_scores=cdn_score.scores,
                vps_coordinates=vps_coordinates,
                probing_budgets=probing_budgets,
                ping_vps_to_target=ping_vps_to_target,
            )

        dump_pickle(
            data=cdn_eval_results,
            output_file=path_settings.RESULTS_PATH
            / f"results_1M_hostnames_{answer_granularity}_per_cdn.pickle",
        )


if __name__ == "__main__":
    asyncio.run(main())
