import random
import numpy as np
from loguru import logger
from multiprocessing import Pool

from common.files_utils import load_json
from common.geoloc import (
    haversine,
    select_best_guess_centroid,
    is_within_cirle,
    polygon_centroid,
    circle_intersections,
    circle_preprocessing,
)

from common.settings import ConstantSettings

constant = ConstantSettings()


def compute_rtts_per_dst_src(table, filter, threshold, is_per_prefix=False):
    """
    Compute the guessed geolocation of the targets
    """
    clickhouse_wrapper = Clickhouse(
        host=CLICKHOUSE_HOST,
        database=CLICKHOUSE_DB,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )

    # Compute the geolocation of the different IP addresses
    if not is_per_prefix:
        query = clickhouse_wrapper.get_min_rtt_per_src_dst_query(
            table, filter=filter, threshold=threshold
        )
    else:
        query = clickhouse_wrapper.get_min_rtt_per_src_dst_prefix_query(
            table, filter=filter, threshold=threshold
        )

    rows = clickhouse_wrapper.execute_iter(query)

    rtt_per_srcs_dst = {}
    for dst, src, min_rtt in rows:
        rtt_per_srcs_dst.setdefault(dst, {})[src] = [min_rtt]

    clickhouse_wrapper.client.disconnect()

    return rtt_per_srcs_dst


def compute_error(dst, vp_coordinates_per_ip, rtt_per_src):
    error = None
    circles = []
    guessed_geolocation_circles = select_best_guess_centroid(
        dst, vp_coordinates_per_ip, rtt_per_src
    )
    if guessed_geolocation_circles is not None:
        guessed_geolocation, circles = guessed_geolocation_circles
        real_geolocation = vp_coordinates_per_ip[dst]
        error = haversine(guessed_geolocation, real_geolocation)
    return error, circles


def round_based_algorithm_impl(
    dst,
    rtt_per_src,
    vp_coordinates_per_ip,
    vps_per_target_greedy,
    asn_per_vp,
    threshold,
):
    # Only take the first n_vps
    vp_coordinates_per_ip_allowed = {
        x: vp_coordinates_per_ip[x]
        for x in vp_coordinates_per_ip
        if x in vps_per_target_greedy
    }
    guessed_geolocation_circles = select_best_guess_centroid(
        dst, vp_coordinates_per_ip_allowed, rtt_per_src
    )
    if guessed_geolocation_circles is None:
        return dst, None, None
    guessed_geolocation, circles = guessed_geolocation_circles
    # Then take one probe per AS, city in the zone
    probes_in_intersection = {}
    for probe, probe_coordinates in vp_coordinates_per_ip.items():
        is_in_intersection = True
        for circle in circles:
            lat_c, long_c, rtt_c, d_c, r_c = circle
            if not is_within_cirle(
                (lat_c, long_c), rtt_c, probe_coordinates, speed_threshold=2 / 3
            ):
                is_in_intersection = False
                break
        if is_in_intersection:
            probes_in_intersection[probe] = probe_coordinates

    # Now only take one probe per AS/city in the probes in intersection
    selected_probes_per_asn = {}
    for probe in probes_in_intersection:
        asn = asn_per_vp[probe]
        if asn not in selected_probes_per_asn:
            selected_probes_per_asn.setdefault(asn, []).append(probe)
            continue
        else:
            is_already_found_close = False
            for selected_probe in selected_probes_per_asn[asn]:
                distance = haversine(
                    vp_coordinates_per_ip[probe], vp_coordinates_per_ip[selected_probe]
                )
                if distance < threshold:
                    is_already_found_close = True
                    break
            if not is_already_found_close:
                # Add this probe to the selected as we do not already have the same probe.
                selected_probes_per_asn[asn].append(probe)

    selected_probes = set()
    for _, probes in selected_probes_per_asn.items():
        selected_probes.update(probes)

    vp_coordinates_per_ip_tier2 = {
        x: vp_coordinates_per_ip[x]
        for x in vp_coordinates_per_ip
        if x in selected_probes
    }
    vp_coordinates_per_ip_tier2[dst] = vp_coordinates_per_ip[dst]
    # Now evaluate the error with this subset of probes
    error, circles = compute_error(dst, vp_coordinates_per_ip_tier2, rtt_per_src)
    return dst, error, len(selected_probes)


def round_based_algorithm(
    greedy_probes, rtt_per_srcs_dst, vp_coordinates_per_ip, asn_per_vp, n_vps, threshold
):
    """
    First is to use a subset of greedy probes, and then take 1 probe/AS in the given CBG area
    :param greedy_probes:
    :return:
    """

    vps_per_target_greedy = set(greedy_probes[:n_vps])

    args = []
    for i, (dst, rtt_per_src) in enumerate(sorted(rtt_per_srcs_dst.items())):
        if dst not in vp_coordinates_per_ip:
            continue
        args.append(
            (
                dst,
                rtt_per_src,
                vp_coordinates_per_ip,
                vps_per_target_greedy,
                asn_per_vp,
                threshold,
            )
        )
    with Pool(24) as p:
        results = p.starmap(round_based_algorithm_impl, args)
        return results


if __name__ == "__main__":
    logger.info("IMC baseline testing")

    greedy_probes = load_json(GREEDY_PROBES_FILE)
    rtt_per_srcs_dst = compute_rtts_per_dst_src(
        PROBES_TO_ANCHORS_PING_TABLE, filter, threshold=100
    )
    vp_distance_matrix = load_json(PAIRWISE_DISTANCE_FILE)

    error_cdf = round_based_algorithm(
        greedy_probes,
        rtt_per_srcs_dst,
        vp_coordinates_per_ip,
        asn_per_vp_ip,
        tier1_vps,
        threshold=40,
    )
    error_cdf_per_tier1_vps[tier1_vps] = error_cdf
