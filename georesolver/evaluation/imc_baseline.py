import math

from multiprocessing import Pool, cpu_count
from loguru import logger
from pathlib import Path

from georesolver.common.files_utils import load_json, dump_json
from georesolver.clickhouse.queries import get_pings_per_src_dst, load_vps, load_targets
from georesolver.common.geoloc import (
    select_best_guess_centroid,
    haversine,
    is_within_cirle,
    distance,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def imc_baseline_measurement_cost() -> None:
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)

    imc_baseline_results = load_json(
        path_settings.RESULTS_PATH / "round_based_algo_file.json"
    )

    imc_cost = sum([c[1] for c in imc_baseline_results[500]])

    logger.info("Measurement cost:: ")
    logger.info(f"All VPs: {len(vps) * len(targets)}")
    logger.info(f"IMC Baseline: {imc_cost}")


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
    compute_distance: bool = True,
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
                d = haversine(
                    vp_coordinates_per_ip[probe], vp_coordinates_per_ip[selected_probe]
                )
                if d < threshold:
                    is_already_found_close = True
                    break
            if not is_already_found_close:
                # Add this probe to the selected as we do not already have the same probe.
                selected_probes_per_asn[asn].append(probe)

    selected_probes = set()
    for _, probes in selected_probes_per_asn.items():
        selected_probes.update(probes)

    rtts = []
    for probe in selected_probes:
        try:
            min_rtt = rtt_per_src[probe]
            rtts.append((probe, min_rtt))
        except KeyError:
            continue

    if not rtts:
        logger.info(f"No ping available for :: {dst}")
        return (dst, -1, -1, -1)

    shortest_ping_vp_addr, shortest_ping_rtt = min(rtts, key=lambda x: x[-1])
    shortest_ping_lat, shortest_ping_lon = vp_coordinates_per_ip[shortest_ping_vp_addr]

    if compute_distance:
        target_lat, target_lon = vp_coordinates_per_ip[dst]
        error = distance(target_lat, shortest_ping_lat, target_lon, shortest_ping_lon)
    else:
        error = -1

    return (
        dst,
        error,
        len(vps_per_target_greedy) + len(selected_probes),
        shortest_ping_rtt,
    )


def round_based_algorithm(
    greedy_probes,
    rtt_per_srcs_dst,
    vp_coordinates_per_ip,
    asn_per_vp,
    n_vps,
    threshold,
    compute_dist: bool = True,
):
    """
    First is to use a subset of greedy probes, and then take 1 probe/AS in the given CBG area
    :param greedy_probes:
    :return:
    """

    vps_per_target_greedy = set(greedy_probes[:n_vps])

    args = []
    for _, (dst, rtt_per_src) in enumerate(sorted(rtt_per_srcs_dst.items())):
        args.append(
            (
                dst,
                rtt_per_src,
                vp_coordinates_per_ip,
                vps_per_target_greedy,
                asn_per_vp,
                threshold,
                compute_dist,
            )
        )

    usable_cpu = cpu_count() - 1
    with Pool(usable_cpu) as p:
        results = p.starmap(round_based_algorithm_impl, args)

    return results


def main(
    targets: list[str],
    ping_table: str,
    output_path: str,
    compute_dist: bool = True,
) -> None:
    if not (path_settings.GREEDY_VPS).exists():
        vp_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)
        greedy_vp_selection(vp_distance_matrix)

    greedy_probes = load_json(path_settings.GREEDY_VPS)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)

    rtt_per_srcs_dst = get_pings_per_src_dst(
        table_name=ping_table,
        removed_vps=removed_vps,
    )

    # filter pings
    filtered_rtt_per_srcs_dst = {}
    for dst_addr, rtt_per_srcs in rtt_per_srcs_dst.items():
        if dst_addr not in targets:
            continue

        filtered_rtt_per_srcs_dst[dst_addr] = rtt_per_srcs

    vp_coordinates = {}
    asn_per_vp_ip = {}
    for vp in vps:
        asn_v4 = vp["asn_v4"]
        asn_per_vp_ip[vp["addr"]] = vp["asn_v4"]
        vp_coordinates[vp["addr"]] = vp["lat"], vp["lon"]

    error_cdf_per_tier1_vps = {}
    for tier1_vps in [500]:
        logger.info(f"Using {tier1_vps} tier1_vps")
        error_cdf = round_based_algorithm(
            greedy_probes,
            filtered_rtt_per_srcs_dst,
            vp_coordinates,
            asn_per_vp_ip,
            tier1_vps,
            threshold=40,
            compute_dist=compute_dist,
        )
        error_cdf_per_tier1_vps[tier1_vps] = error_cdf

    dump_json(
        error_cdf_per_tier1_vps,
        output_path,
    )


def measurement_overhead(input_path: Path) -> int:
    results = load_json(input_path)
    overall_cost = 0
    for _, results_per_target in results.items():
        logger.info(f"NB targets:: {len(results_per_target)}")
        for result in results_per_target:
            if result[2]:
                overall_cost += result[2]

    return overall_cost


if __name__ == "__main__":
    # 1. measurement overhead for anchors
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    targets = [target["addr"] for target in targets]
    ping_table = clickhouse_settings.VPS_VPS_MESHED_PINGS_TABLE
    output_path = path_settings.RESULTS_PATH / "imc_baseline_anchors_dataset.json"

    if not output_path.exists():
        main(
            targets=targets,
            ping_table=ping_table,
            compute_dist=False,
            output_path=output_path,
        )

    overhead = measurement_overhead(output_path)
    logger.info(f"ANCHORS dataset:: {overhead*3} pings overhead")

    # 2. measurement overhead for ripe ip map
    targets = load_json(
        path_settings.END_TO_END_DATASET / "ripe_ip_map_targets_evaluation.json"
    )
    ping_table = "pings_end_to_end"
    output_path = path_settings.RESULTS_PATH / "imc_baseline_ripe_ip_map_dataset.json"
    if not output_path.exists():
        main(
            targets=targets,
            ping_table=ping_table,
            compute_dist=False,
            output_path=output_path,
        )

    overhead = measurement_overhead(output_path)
    logger.info(f"RIPE IP MAP dataset:: {overhead* 3} pings overhead")

    # 3. measurement overhead for routers
    targets = load_json(
        path_settings.END_TO_END_DATASET / "routers_targets_evaluation.json"
    )
    ping_table = "pings_end_to_end"
    output_path = path_settings.RESULTS_PATH / "imc_baseline_routers_dataset.json"
    if not output_path.exists():
        main(
            targets=targets,
            ping_table=ping_table,
            compute_dist=False,
            output_path=output_path,
        )

    overhead = measurement_overhead(output_path)
    logger.info(f"ROUTERS dataset:: {overhead * 3} pings overhead")
