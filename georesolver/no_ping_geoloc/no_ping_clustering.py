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

from geogiant.common.queries import (
    get_pings_per_target,
    load_targets,
    load_vps,
)
from geogiant.ecs_vp_selection.utils import (
    get_ecs_pings,
    select_one_vp_per_as_city,
    get_parsed_vps,
)
from geogiant.common.utils import TargetScores
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


def get_ecs_vps(
    target_subnet: str,
    target_score: dict,
    vps_per_subnet: dict,
    probing_budget: int = 50,
) -> list:
    """
    get the target score and extract best VPs function of the probing budget
    return 1 VP per subnet, TODO: get best connected VP per subnet
    """
    # retrieve all vps belonging to subnets with highest mapping scores
    ecs_vps = []
    for subnet, score in target_score[:probing_budget]:
        vps_in_subnet = vps_per_subnet[subnet]

        if subnet == target_subnet:
            continue

        if not vps_in_subnet:
            continue

        ecs_vps.append((vps_in_subnet[0], score))

    return ecs_vps


def filter_vps_per_cluster(
    vps_per_cluster: dict,
    vps_coordinates: dict,
    pairwise_distance: dict,
    nb_vps_per_cluster: int = 10,
) -> list:
    """
    return the per cluster ecs vps selection
    maximize AS diversity for VP in cluster,
    select also noisy vps, TODO: remove some noise
    """
    selected_vps = []
    for label, vps in vps_per_cluster.items():
        if label == -1:
            # select outliers if they are alone in their area
            close_vps = []
            for vp in vps:
                vp_pairwise_distance = pairwise_distance[vp[0]]
                for neighbors, distance in vp_pairwise_distance:
                    if distance < 40:  # we take 40km for "city level closeness"
                        close_vps.append(neighbors)

                if len(close_vps) < 5:
                    selected_vps.append(vp[0])

            selected_vps.extend([vp[0] for i, vp in enumerate(vps)])

        else:
            cluster_vps = []
            cluster_as = set()
            vps_per_as = defaultdict(list)
            for vp_addr, vp_lat, vp_lon, score in vps:
                _, _, vp_asn = vps_coordinates[vp_addr]

                vps_per_as[vp_asn].append((vp_addr, vp_lat, vp_lon, score))

                # # take on VP per AS in the cluster
                # if vp_asn not in cluster_as:
                #     cluster_as.add(vp_asn)
                #     cluster_vps.append([vp_addr, vp_lat, vp_lon, score])

            # take one VP per AS, max score per AS
            for asn, vp_info in vps_per_as.items():
                cluster_vps.append(max(vp_info, key=lambda x: x[-1]))

            # cluster_vps = sorted(cluster_vps, key=lambda x: x[-1], reverse=True)
            # cluster_vps = cluster_vps[:nb_vps_per_cluster]

            # only take VPs with maximum score within the cluster
            selected_vps.extend(cluster_vps)

    return selected_vps


def get_all_vps_per_cluster(cluster_points: np.array, vps_coordinates: dict) -> list:
    """return all VPs present within a given cluster"""
    all_vps_in_cluster = set()
    cluster_polygon = Polygon(cluster_points)

    for vp_addr, (vp_lat, vp_lon, _) in vps_coordinates.items():
        point = Point((vp_lat, vp_lon))

        if point.within(cluster_polygon):
            all_vps_in_cluster.add((vp_addr, vp_lat, vp_lon))

    return all_vps_in_cluster


def shortest_ping(selected_vps: list, pings: dict) -> tuple:
    """return the shortest ping for a selection of VPs and a set of measurements"""
    try:
        selected_pings = get_ecs_pings(
            target_associated_vps=selected_vps,
            ping_to_target=pings,
        )
    except KeyError:
        return None

    if not selected_pings:
        return None, None

    addr, min_rtt = min(selected_pings, key=lambda x: x[-1])

    return addr, min_rtt


def get_vp_score(vp_subnet: str, target_score: list) -> tuple[float, float]:
    """retrieve vp score"""
    score = -1
    index = -1
    for i, (subnet, score) in enumerate(target_score):
        if vp_subnet == subnet:
            score = score
            index = i
            break

    return score, index


def get_vp_info(
    target: dict,
    target_score: list,
    vp_addr: str,
    vps_coordinates: dict,
    rtt: float = None,
) -> dict:
    """get all useful information about a selected VP"""
    vp_subnet = get_prefix_from_ip(vp_addr)
    lat, lon, _ = vps_coordinates[vp_addr]
    d_error = distance(target["lat"], lat, target["lon"], lon)
    score, index = get_vp_score(vp_subnet, target_score)

    return {
        "addr": vp_addr,
        "subnet": vp_subnet,
        "lat": lat,
        "lon": lon,
        "rtt": rtt,
        "d_error": d_error,
        "score": score,
        "index": index,
    }


def get_no_ping_vp(
    target, target_score: list, vps_per_subnet: dict, vps_coordinates: dict
) -> dict:
    """return VP with maximum score"""
    subnet, _ = target_score[1:]
    vp_addr = vps_per_subnet[subnet][0]

    return get_vp_info(target, target_score, vp_addr, vps_coordinates)


def get_cluster_scores_fraction(vps_per_cluster: dict, vps_coordinates: dict) -> dict:
    """
    compute a score for each cluster
    score: number of
    """
    cluster_scores = {}
    for label, vps in vps_per_cluster.items():
        # do not consider noisy points
        if label == -1:
            continue

        # avoid small clusters
        if len(vps) < 3:
            continue

        cluster_poly, _ = find_exterior_points([(vp[1], vp[2]) for vp in vps])
        all_vps_per_cluster = get_all_vps_per_cluster(cluster_poly, vps_coordinates)

        # just in case, add vps that were already in the cluster
        all_vps_per_cluster = set(all_vps_per_cluster)
        for vp in vps:
            all_vps_per_cluster.add(vp[0])

        # we cannot find vps in cluster polygon?
        if not all_vps_per_cluster:
            continue

        # get the cluster score
        cluster_score = len(vps) / len(all_vps_per_cluster)
        cluster_scores[label] = cluster_score

    return cluster_scores


def get_cluster_scores_avg(ecs_vps_per_cluster: dict, vps_coordinates: dict) -> dict:
    """
    get the cluster that has the best ECS avg score
    """
    cluster_scores = {}
    for label, vps in ecs_vps_per_cluster.items():
        if label == -1:
            continue

        # avoid small clusters
        if len(vps) < 3:
            continue

        cluster_poly, _ = find_exterior_points([(vp[1], vp[2]) for vp in vps])
        all_vps_per_cluster = get_all_vps_per_cluster(cluster_poly, vps_coordinates)

        # just in case, add vps that were already in the cluster
        all_vps_per_cluster = set(all_vps_per_cluster)
        for vp in vps:
            all_vps_per_cluster.add(vp)

        cluster_scores[label] = np.mean([vp[-1] for vp in all_vps_per_cluster])

    return cluster_scores


def get_no_ping_cluster_vp(
    target: dict,
    target_score: list,
    ecs_vps_per_cluster: dict,
    vps_coordinates: dict,
) -> dict:
    """
    Select VP with highest score per cluster, return VP of the main cluster
    (i.e. cluster that have the highest score)
    """
    cluster_scores = get_cluster_scores_fraction(ecs_vps_per_cluster, vps_coordinates)

    if not cluster_scores:
        return None

    best_cluster_vps = []
    best_cluster_label, _ = max(cluster_scores.items(), key=lambda x: x[1])
    best_cluster_vps = ecs_vps_per_cluster[best_cluster_label]

    if best_cluster_vps:
        max_score_vp_best_cluster = max(best_cluster_vps, key=lambda x: x[-1])
        return get_vp_info(
            target=target,
            target_score=target_score,
            vp_addr=max_score_vp_best_cluster[0],
            vps_coordinates=vps_coordinates,
            rtt=None,
        )
    else:
        return None


def parse_target(target: dict, asndb: pyasn) -> dict:
    """simply get target into a nice dict structure"""
    addr = target["addr"]
    subnet = get_prefix_from_ip(addr)
    bgp_prefix = route_view_bgp_prefix(subnet, asndb)

    return {
        "addr": addr,
        "subnet": subnet,
        "bgp_prefix": bgp_prefix,
        "lat": target["lat"],
        "lon": target["lon"],
    }


def ecs_dns_vp_selection_eval(
    targets: list,
    vps_per_subnet: dict,
    subnet_scores: dict,
    ping_vps_to_target: dict,
    vps_coordinates: dict,
    pairwise_distance: dict,
) -> tuple[dict, dict]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    for target in tqdm(targets):

        target = parse_target(target, asndb)

        try:
            target_score = subnet_scores[target["subnet"]]
            target_score = target_score["jaccard"]
        except KeyError:
            logger.error(f"cannot find target score for subnet : {target['subnet']}")

        ecs_vps = get_ecs_vps(target["subnet"], target_score, vps_per_subnet, 50)

        # CLUSTER SELECTION
        clusters, n_clusters, n_noise = get_cluster(ecs_vps, vps_coordinates)

        ecs_vps_per_cluster = get_vp_in_cluster(clusters, ecs_vps, vps_coordinates)

        # Filter VPs based on clusters (N VPs max per cluster / remove some outliers)
        filtered_vps_per_cluster = filter_vps_per_cluster(
            ecs_vps_per_cluster, vps_coordinates, pairwise_distance, 5
        )

        no_ping_cluster_vp = get_no_ping_cluster_vp(
            target,
            target_score,
            ecs_vps_per_cluster,
            vps_coordinates,
        )

        results[target["addr"]] = {
            "target": target,
            "no_ping_cluster_vp": no_ping_cluster_vp,
            "ecs_scores": target_score[:50],
            "ecs_vps": ecs_vps,
            "ecs_vps_per_cluster": ecs_vps_per_cluster,
            "filtered_vps_per_cluster": filtered_vps_per_cluster,
            "n_cluster": n_clusters,
            "n_noise": n_noise,
        }

    return results


def get_metrics(
    targets: list,
    vps_per_subnet: dict,
    subnet_scores: dict,
    vps_coordinates: dict,
    ping_vps_to_target: dict,
    pairwise_distance: dict,
) -> dict:
    """get geolocation error function of the probing budget"""
    results = {}

    results = ecs_dns_vp_selection_eval(
        targets=targets,
        vps_per_subnet=vps_per_subnet,
        subnet_scores=subnet_scores,
        vps_coordinates=vps_coordinates,
        ping_vps_to_target=ping_vps_to_target,
        pairwise_distance=pairwise_distance,
    )

    return results


async def main() -> None:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    pairwise_distance = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.VPS_VPS_MESHED_PINGS_TABLE
    )
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)

    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)

    logger.info("BGP prefix score geoloc evaluation")

    subnet_scores: TargetScores = load_pickle(
        path_settings.RESULTS_PATH
        / "tier4_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    )

    eval_results = get_metrics(
        targets=targets,
        vps_per_subnet=vps_per_subnet,
        subnet_scores=subnet_scores.score_answer_subnets,
        vps_coordinates=vps_coordinates,
        ping_vps_to_target=ping_vps_to_target,
        pairwise_distance=pairwise_distance,
    )

    dump_pickle(
        data=eval_results,
        output_file=path_settings.RESULTS_PATH
        / f"evaluation_1M_hostnames_answer_bgp_prefix_max_bgp_prefix_clusters.pickle",
    )


if __name__ == "__main__":
    asyncio.run(main())
