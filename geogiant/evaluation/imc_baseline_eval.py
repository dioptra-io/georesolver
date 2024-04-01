import math

from multiprocessing import Pool, cpu_count
from loguru import logger

from geogiant.common.files_utils import load_json, dump_json
from geogiant.common.queries import get_pings_per_src_dst, load_vps
from geogiant.common.geoloc import (
    select_best_guess_centroid,
    haversine,
    is_within_cirle,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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
    for _, (dst, rtt_per_src) in enumerate(sorted(rtt_per_srcs_dst.items())):
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

    usable_cpu = cpu_count() - 1
    with Pool(usable_cpu) as p:
        results = p.starmap(round_based_algorithm_impl, args)
        return results


def greedy_selection_probes_impl(probe, distance_per_probe, selected_probes):
    distances_log = [
        math.log(distance_per_probe[p])
        for p in selected_probes
        if p in distance_per_probe and distance_per_probe[p] > 0
    ]
    total_distance = sum(distances_log)
    return probe, total_distance


def greedy_vp_selection(vp_distance_matrix: dict) -> None:
    """select VPs with greedy selection to cover maximum coverage"""
    logger.info("Starting greedy algorithm")

    selected_probes = []
    remaining_probes = set(vp_distance_matrix.keys())
    usable_cpu = cpu_count() - 1
    with Pool(usable_cpu) as p:
        while len(remaining_probes) > 0 and len(selected_probes) < 1_000:
            args = []
            for probe in remaining_probes:
                args.append((probe, vp_distance_matrix[probe], selected_probes))

            results = p.starmap(greedy_selection_probes_impl, args)

            furthest_probe_from_selected, _ = max(results, key=lambda x: x[1])
            selected_probes.append(furthest_probe_from_selected)
            remaining_probes.remove(furthest_probe_from_selected)

    dump_json(selected_probes, path_settings.GREEDY_VPS)


if __name__ == "__main__":

    if not (path_settings.GREEDY_VPS).exists():
        vp_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)
        greedy_vp_selection(vp_distance_matrix)

    greedy_probes = load_json(path_settings.GREEDY_VPS)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    removed_vps = load_json(path_settings.REMOVED_VPS)

    # clickhouse is required here
    rtt_per_srcs_dst = get_pings_per_src_dst(
        clickhouse_settings.PING_VPS_TO_TARGET, threshold=100
    )

    vp_coordinates = {}
    asn_per_vp_ip = {}
    for vp in vps:
        asn_v4 = vp["asn_v4"]
        asn_per_vp_ip[vp["addr"]] = vp["asn_v4"]
        vp_coordinates[vp["addr"]] = vp["lat"], vp["lon"]

    error_cdf_per_tier1_vps = {}
    for tier1_vps in [10, 100, 300, 500, 1000]:
        logger.info(f"Using {tier1_vps} tier1_vps")
        error_cdf = round_based_algorithm(
            greedy_probes,
            rtt_per_srcs_dst,
            vp_coordinates,
            asn_per_vp_ip,
            tier1_vps,
            threshold=40,
        )
        error_cdf_per_tier1_vps[tier1_vps] = error_cdf

    dump_json(error_cdf_per_tier1_vps, ROUND_BASED_ALGORITHM_FILE)
