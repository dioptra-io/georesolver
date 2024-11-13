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
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

CONFIG_PATH = (
    path_settings.EXPERIMENT_PATH
    / "subnet_aggregation__78uy56er-cd58-4ff9-ad7f-41fa8ad26a3f.json"
)
PING_TABLE = "ecs_aggregation_ping"


def load_pings_per_subnet() -> dict:
    """retrieve all pings per IP address, group by prefix"""
    pings_per_subnet = defaultdict(list)
    ping_per_targets = get_pings_per_target(PING_TABLE)

    for target_addr, pings in ping_per_targets.items():
        subnet = get_prefix_from_ip(target_addr)
        pings_per_subnet[subnet].append((target_addr, pings))

    return pings_per_subnet


def get_frac_per_subnet(
    pings_per_subnet: dict[list], latency_threshold: int = 2
) -> list:
    """calculate the fraction of IP addresses with a latency under 2ms"""
    frac_per_subnet = []
    frac_per_subnet_filtered = []
    for _, pings_per_target in pings_per_subnet.items():

        # get the shortest ping per target
        shortest_pings_per_ip = []
        shortest_pings_per_ip_filtered = []
        for _, pings in pings_per_target:
            _, shortest_ping = min(pings, key=lambda x: x[-1])
            shortest_pings_per_ip.append(shortest_ping)

            if shortest_ping < 2:
                shortest_pings_per_ip_filtered.append(shortest_ping)

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
    for subnet, pings_per_target in pings_per_subnet.items():
        if len(pings_per_target) < 3:
            continue

        # extract N random targets from subnet results
        representatives = sample(pings_per_target, nb_representatives)
        representatives_addr = [target for target, _ in representatives]

        # get median latency per vps
        reduced_vps = get_reduced_vps(representatives, nb_vps)

        # get min RTT for the rest of the targets
        reduced_pings = []
        for target_addr, pings in pings_per_target:
            if target_addr in representatives_addr:
                continue

            for vp_addr, min_rtt in pings:
                if vp_addr not in reduced_vps:
                    continue

                reduced_pings.append((vp_addr, min_rtt))

            try:
                georesolver_geoloc = min(pings, key=lambda x: x[-1])
                reduced_geoloc = min(reduced_pings, key=lambda x: x[-1])
            except ValueError:
                logger.error(f"Cannot find reduced pings for addr {target_addr}")

            # get georesolver and reduced vps selection results
            georesolver_results[target_addr] = georesolver_geoloc
            reduced_results[target_addr] = reduced_geoloc

    return georesolver_results, reduced_results


##########################################################################################
# PLOT PART
##########################################################################################


def main() -> None:
    """
    we ought to evaluate three thinks:
        1. The fraction of IP addresses we geolocate within the same area within each /24
        (using latency and elected VP)
        2. Evaluate the same metric but for each AS type (as we did for continental geoloc)
        3. Re-evaluate the results described in the Million scale paper
    """
    do_fraction_subnet_eval: bool = False
    do_per_AS_eval: bool = True
    do_reduced_eval: bool = True

    pings_per_subnet = load_pings_per_subnet()

    if do_fraction_subnet_eval:
        cdfs = []
        frac_per_subnet, frac_per_subnet_filtered = get_frac_per_subnet(
            pings_per_subnet, 2
        )
        x, y = ecdf(frac_per_subnet)
        cdfs.append((x, y, "fraction per subnet"))

        x, y = ecdf(frac_per_subnet_filtered)
        cdfs.append((x, y, "fraction per subnet (filtered)"))

        plot_multiple_cdf(
            cdfs=cdfs,
            output_path="subnet_aggregation__fraction_under_2ms",
            metric_evaluated="",
            x_limit_left=0,
            y_label="CDF of subnets",
            x_label="fraction of targets under 2ms",
            x_log_scale=False,
        )

    if do_reduced_eval:
        georesolver_results, reduced_results = get_reduced_results(
            pings_per_subnet, 3, 10
        )

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

        # TODO: latency diff between georesolver and reduced

    if do_per_AS_eval:
        pass  # TODO

    # TODO: get ecdf and plot


if __name__ == "__main__":
    main()
