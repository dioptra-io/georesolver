"""plot results for subnet aggregation experiments"""

from numpy import median
from random import sample
from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import load_vps, get_pings_per_target
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    get_proportion_under,
    plot_cdf,
    plot_multiple_cdf,
)
from georesolver.common.geoloc import rtt_to_km, circle_intersections
from georesolver.common.files_utils import load_csv, load_json
from georesolver.common.ip_addresses_utils import get_prefix_from_ip, load_subnet_to_asn
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

TARGET_FILE = (
    path_settings.DATASET / "subnet_aggregation/subnet_aggregation_targets.csv"
)
CONFIG_PATH = (
    path_settings.EXPERIMENT_PATH
    / "subnet_aggregation__78uy56er-cd58-4ff9-ad7f-41fa8ad26a3f.json"
)
PING_TABLE = "subnet_aggregation_ping"


def load_pings_per_subnet(target_subnets: set) -> dict:
    """retrieve all pings per IP address, group by prefix"""
    pings_per_subnet = defaultdict(list)
    ping_per_targets = get_pings_per_target(PING_TABLE)

    for target_addr, pings in ping_per_targets.items():
        subnet = get_prefix_from_ip(target_addr)

        if subnet not in target_subnets:
            continue

        pings_per_subnet[subnet].append((target_addr, pings))

    return pings_per_subnet


def get_frac_per_subnet(
    pings_per_subnet: dict[list], latency_threshold: int = 2
) -> list:
    """calculate the fraction of IP addresses with a latency under 2ms"""
    frac_per_subnet = []
    for _, pings_per_target in pings_per_subnet.items():

        if len(pings_per_target) < 10:
            continue

        # get the shortest ping per target
        shortest_pings_per_ip = []
        for _, pings in pings_per_target:
            _, shortest_ping = min(pings, key=lambda x: x[-1])
            shortest_pings_per_ip.append(shortest_ping)

        # only consider subnet is smallest latency is inferior to 2ms
        if not min(shortest_pings_per_ip) <= 2:
            continue

        # get the proportion under 2 ms
        x, y = ecdf(shortest_pings_per_ip)
        frac = get_proportion_under(x, y, latency_threshold)
        frac_per_subnet.append(frac)

    return frac_per_subnet


def get_frac_per_subnet_per_asn_type(
    pings_per_subnet: dict[list], latency_threshold: int = 2
) -> dict:
    """
    same as getting the fraction of IP address under a certain latency threshold.
    Except this time, results are splitted function of the AS type (CAIDA dataset)
    """
    asn_to_type = load_json(path_settings.STATIC_FILES / "caida_enhanced_AS_types.json")
    subnet_to_asn = load_subnet_to_asn(pings_per_subnet.keys())

    frac_per_subnet_per_as_type = defaultdict(list)
    all_asns = set()
    all_subnets = set()
    for subnet, pings_per_target in pings_per_subnet.items():
        try:
            as_number = subnet_to_asn[subnet]
            as_type = asn_to_type[str(as_number)]
        except KeyError:
            continue

        # get the shortest ping per target
        shortest_pings_per_ip = []
        for _, pings in pings_per_target:
            _, shortest_ping = min(pings, key=lambda x: x[-1])
            shortest_pings_per_ip.append(shortest_ping)

        # only consider subnet is smallest latency is inferior to 2ms
        if not min(shortest_pings_per_ip) <= 2:
            continue

        # get the proportion under 2 ms
        if shortest_pings_per_ip:
            x, y = ecdf(shortest_pings_per_ip)
            frac = get_proportion_under(x, y, latency_threshold)
            frac_per_subnet_per_as_type[as_type].append(frac)

            all_asns.add(as_number)
            all_subnets.add(subnet)

    return frac_per_subnet_per_as_type, len(all_subnets), len(all_asns)


def non_overlapping_geoloc(pings_per_subnet: dict[list], vps_coordinates: dict) -> None:
    """
    For each subnet:
        - get the geolocation for each target
        - check if there exists non-overlapping geolocation
    """
    non_overlapping_geoloc_per_subnet = defaultdict(dict)
    for subnet, pings_per_target in pings_per_subnet.items():

        # first, get each target geoloc and group per VP
        target_geoloc_per_vp = defaultdict(list)
        for target_addr, pings in pings_per_target:
            vp_addr, shortest_ping = min(pings, key=lambda x: x[-1])
            target_geoloc_per_vp[vp_addr].append((target_addr, shortest_ping))

        # check if more than one VP
        if len(target_geoloc_per_vp) <= 1:
            continue

        # more than two VPs. Order each VP's targets per latency
        for vp, targets_geoloc in target_geoloc_per_vp.items():
            target_geoloc_per_vp[vp] = sorted(targets_geoloc, key=lambda x: x[-1])

        # compare geoloc of target with smallest latency with every other
        for vp_i, targets_geoloc_i in target_geoloc_per_vp.items():
            for target_m, min_rtt_m in targets_geoloc_i:
                # compare other geoloc with the smallest circle
                vp_i_lat, vp_i_lon, _, _ = vps_coordinates[vp_i]
                d = rtt_to_km(min_rtt_m)
                circle_i = (vp_i_lat, vp_i_lon, min_rtt_m, d, d / 2)

                # compare targets geoloc with the current one
                for vp_j, targets_geoloc_j in target_geoloc_per_vp.items():
                    if vp_i == vp_j:
                        continue

                    if (vp_i, vp_j) in non_overlapping_geoloc_per_subnet[subnet] or (
                        vp_j,
                        vp_i,
                    ) in non_overlapping_geoloc_per_subnet[subnet]:
                        continue

                    if vp_j in non_overlapping_geoloc_per_subnet[subnet]:
                        continue

                    for target_n, min_rtt_n in targets_geoloc_j:
                        vp_j_lat, vp_j_lon, _, _ = vps_coordinates[vp_j]
                        d = rtt_to_km(min_rtt_n)
                        circle_j = (vp_j_lat, vp_j_lon, min_rtt_n, d, d / 2)

                        intersect, _ = circle_intersections([circle_i, circle_j])

                        # if there is an intersection, no need to check the rest
                        # (we ordered the circles function of the latency)
                        if intersect:
                            break

                        # if the two circles are non-overlapping, save
                        try:
                            non_overlapping_geoloc_per_subnet[subnet][
                                (vp_i, vp_j)
                            ].append((target_m, min_rtt_m, target_n, min_rtt_n))
                        except KeyError:
                            non_overlapping_geoloc_per_subnet[subnet][(vp_i, vp_j)] = [
                                (target_m, min_rtt_m, target_n, min_rtt_n)
                            ]

                # if the two circles described by the target with the smallest
                # latency overlap, no need to continue analysis

                # if no intersection, save subnet non-overlapping targets
                # with their geolocation
                # TODO: improve this step (check with next target, etc.)

        # get the proportion under 2 ms
        x, y = ecdf(shortest_pings_per_ip)
        frac = get_proportion_under(x, y, latency_threshold)
        frac_per_subnet.append(frac)

        if shortest_pings_per_ip_filtered:
            x, y = ecdf(shortest_pings_per_ip)
            frac = get_proportion_under(x, y, latency_threshold)
            frac_per_subnet_filtered.append(frac)

    return frac_per_subnet, frac_per_subnet_filtered


def get_latencies_per_vp(representatives: list) -> dict[list]:
    """return a dictionnary with each VPs shortest ping to 3 representative targets"""
    latencies_per_vp = defaultdict(list)
    for _, pings in representatives:
        for vp_addr, min_rtt in pings:
            latencies_per_vp[vp_addr].append(min_rtt)

    return latencies_per_vp


def get_reduced_vps(representatives: list, nb_vps: int = 10) -> list[str]:
    """get the 10 VPs with the lowest median latency to the representatives IP addrs"""
    latencies_per_vp = get_latencies_per_vp(representatives)

    # get median rtt per VP
    median_latencies_per_vp = []
    for vp_addr, rtts in latencies_per_vp.items():
        median_latencies_per_vp.append((vp_addr, median(rtts)))

    # order by median RTT
    median_latencies_per_vp = sorted(median_latencies_per_vp, key=lambda x: x[-1])

    # take the N VPs with the lowest median latency
    reduced_vps = [vp for vp, _ in median_latencies_per_vp[:nb_vps]]

    return reduced_vps


def get_reduced_results(
    pings_per_subnet: dict[list],
    nb_representatives: int = 3,
    nb_vps: int = 10,
) -> tuple[dict, dict]:
    """
    evaluate the million scale paper results:
        - take three random representative IP addresses per subnet
        - select the 10 VPs with the lowest median error
        - get the results for the rest of the IP addresses
        - compare the results
    """
    georesolver_results = defaultdict(list)
    reduced_results = defaultdict(list)
    for _, pings_per_target in pings_per_subnet.items():
        if len(pings_per_target) < 3:
            continue

        # extract N random targets from subnet results
        representatives = sample(pings_per_target, nb_representatives)
        representatives_addr = [target for target, _ in representatives]

        # get median latency per vps
        reduced_vps = get_reduced_vps(representatives, nb_vps)

        # get min RTT for the rest of the targets
        for target_addr, pings in pings_per_target:
            if target_addr in representatives_addr:
                continue

            reduced_pings = []
            for vp_addr, min_rtt in pings:
                if vp_addr not in reduced_vps:
                    continue

                reduced_pings.append((vp_addr, min_rtt))

            try:
                georesolver_geoloc = min(pings, key=lambda x: x[-1])
                reduced_geoloc = min(reduced_pings, key=lambda x: x[-1])
            except ValueError:
                logger.error(f"Cannot find reduced pings for addr {target_addr}")
                continue

            # get georesolver and reduced vps selection results
            georesolver_results[target_addr] = georesolver_geoloc
            reduced_results[target_addr] = reduced_geoloc

    return georesolver_results, reduced_results


##########################################################################################
# PLOT PART
##########################################################################################
def plot_latency_diff(georesolver_results: dict, reduced_results: dict) -> None:
    latency_diffs = []
    for target_addr, georesolver_geoloc in georesolver_results.items():
        try:
            reduced_geoloc = reduced_results[target_addr]
        except KeyError:
            continue

        latency_diff = georesolver_geoloc[-1] - reduced_geoloc[-1]
        latency_diffs.append(latency_diff)

    x, y = ecdf(latency_diffs)
    plot_cdf(
        x=x,
        y=y,
        output_path="subnet_aggregation__million_scale_reevaluation",
        x_label="latency difference",
        y_label="CDF of targets",
        x_lim=0.01,
        x_log_scale=True,
    )


def main() -> None:
    """
    we ought to evaluate three thinks:
        1. The fraction of IP addresses we geolocate within the same area within each /24
        (using latency and elected VP)
        2. Evaluate the same metric but for each AS type (as we did for continental geoloc)
        3. Re-evaluate the results described in the Million scale paper
    """
    do_fraction_subnet_eval: bool = True
    do_reduced_eval: bool = True
    do_per_asn_eval: bool = True

    targets = load_csv(TARGET_FILE)
    target_subnets = set([get_prefix_from_ip(t) for t in targets])
    pings_per_subnet = load_pings_per_subnet(target_subnets)

    if do_fraction_subnet_eval:
        cdfs = []
        frac_per_subnet = get_frac_per_subnet(pings_per_subnet, 2)
        x, y = ecdf(frac_per_subnet)
        cdfs.append((x, y, "fraction per subnet"))

        plot_multiple_cdf(
            cdfs=cdfs,
            output_path="subnet_aggregation__fraction_under_2ms",
            metric_evaluated="",
            x_limit_left=0,
            y_label="CDF of subnets",
            x_label="fraction of targets under 2ms",
            x_log_scale=True,
        )

    if do_reduced_eval:
        georesolver_results, reduced_results = get_reduced_results(
            pings_per_subnet, 3, 10
        )

        georesolver_latencies = [g[-1] for g in georesolver_results.values()]
        reduced_latencies = [g[-1] for g in reduced_results.values()]

        logger.info(f"Nb targets georesolver :: {len(georesolver_results)}")
        logger.info(f"Nb targets reduced     :: {len(reduced_results)}")

        cdfs = []
        x, y = ecdf(georesolver_latencies)
        cdfs.append((x, y, "georesolver"))
        georesolver_under_2_ms = get_proportion_under(x, y, 2)

        x, y = ecdf(reduced_latencies)
        cdfs.append((x, y, "reduced"))
        reduced_under_2_ms = get_proportion_under(x, y, 2)

        plot_multiple_cdf(
            cdfs=cdfs,
            output_path="subnet_aggregation__geo_vs_reduced",
            metric_evaluated="rtt",
            y_label="CDF of targets",
            x_label="min rtt",
            x_log_scale=True,
        )

        logger.info(f"Under 2ms geoRes  :: {georesolver_under_2_ms * 100}[%]")
        logger.info(f"Under 2ms reduced :: {reduced_under_2_ms * 100}[%]")

    if do_per_asn_eval:
        frac_per_subnet_per_AS_type, nb_subnets, nb_asn = (
            get_frac_per_subnet_per_asn_type(
                pings_per_subnet=pings_per_subnet, latency_threshold=2
            )
        )

        cdfs = []
        for as_type, frac_per_subnet in frac_per_subnet_per_AS_type.items():
            x, y = ecdf(frac_per_subnet)
            fraction_of_subnets = round((len(frac_per_subnet) / nb_subnets) * 100, 2)
            cdfs.append(
                (
                    x,
                    y,
                    f"fraction per subnet, {as_type}, {fraction_of_subnets}[%] subnets",
                )
            )

        plot_multiple_cdf(
            cdfs=cdfs,
            output_path="subnet_aggregation__fraction_under_2ms_per_as_type",
            metric_evaluated="",
            x_limit_left=0,
            y_label="CDF of subnets",
            x_label="fraction of targets under 2ms",
            x_log_scale=False,
        )


if __name__ == "__main__":
    main()
