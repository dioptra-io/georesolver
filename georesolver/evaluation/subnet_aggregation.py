"""plot results for subnet aggregation experiments"""

from numpy import median
from random import sample
from loguru import logger
from tqdm import tqdm
from pyasn import pyasn
from collections import defaultdict

from georesolver.clickhouse.queries import load_vps, get_pings_per_target_extended
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    get_proportion_under,
    get_proportion_over,
    plot_cdf,
    plot_multiple_cdf,
)
from georesolver.common.utils import get_parsed_vps
from georesolver.common.geoloc import rtt_to_km, distance
from georesolver.common.files_utils import (
    load_csv,
    load_json,
    load_anycatch_data,
    dump_pickle,
)
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    load_subnet_to_asn,
    route_view_bgp_prefix,
)
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
RESULT_PATH = path_settings.RESULTS_PATH / "subnet_aggregation"


def load_pings_per_subnet(target_subnets: set) -> dict:
    """retrieve all pings per IP address, group by prefix"""
    pings_per_subnet = defaultdict(list)
    ping_per_targets = get_pings_per_target_extended(PING_TABLE)
    anycatch_db = load_anycatch_data()
    asndb = pyasn(str(path_settings.RIB_TABLE))

    for target_addr, pings in ping_per_targets.items():
        subnet = get_prefix_from_ip(target_addr)
        _, bgp_prefix = route_view_bgp_prefix(target_addr, asndb)

        # remove anycast prefix
        if bgp_prefix in anycatch_db:
            continue

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
            _, _, shortest_ping = min(pings, key=lambda x: x[-1])
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
            _, _, shortest_ping = min(pings, key=lambda x: x[-1])
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
    """for each subnet, find if some IP addresses were geolocated in different metro"""
    # TODO
    problematic_geoloc = defaultdict(dict)
    for subnet, pings_per_target in tqdm(pings_per_subnet.items()):
        shortest_ping_per_target = {}

        rtts = []
        for target_addr, pings in pings_per_target:
            vp_addr, vp_id, min_rtt = min(pings, key=lambda x: x[-1])
            shortest_ping_per_target[target_addr] = (vp_addr, vp_id, min_rtt)
            rtts.append(min_rtt)

        if min(rtts) > 2:
            continue

        for target_m, (
            vp_i_addr,
            vp_i_id,
            min_rtt_i,
        ) in shortest_ping_per_target.items():
            for target_n, (
                vp_j_addr,
                vp_j_id,
                min_rtt_j,
            ) in shortest_ping_per_target.items():
                if target_m == target_n:
                    continue

                if vp_i_id == vp_j_id:
                    continue

                # check that the two circles intersect
                # vp_i = vps_per_id[vp_i_id]
                # vp_j = vps_per_id[vp_j_id]

                # vp_i_lat, vp_i_lon = vp_i["lat"], vp_i["lon"]
                # vp_j_lat, vp_j_lon = vp_j["lat"], vp_j["lon"]
                try:
                    vp_i_lat, vp_i_lon, _, _ = vps_coordinates[vp_i_addr]
                    vp_j_lat, vp_j_lon, _, _ = vps_coordinates[vp_j_addr]
                except KeyError:
                    continue

                # get all distances
                d_i = rtt_to_km(min_rtt_i)
                d_j = rtt_to_km(min_rtt_j)
                d_vp_i_to_vp_j = distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

                # speed of light violation test
                if d_i + d_j < d_vp_i_to_vp_j:
                    print(f"{subnet=}")
                    print(f"{target_m=}")
                    print(f"{target_n=}")
                    print(f"{vp_i_addr=}")
                    print(f"{vp_j_addr=}")
                    print(f"{min_rtt_i=}")
                    print(f"{min_rtt_j=}")
                    print(f"{d_i=}")
                    print(f"{d_j=}")
                    print(f"{d_vp_i_to_vp_j=}")
                    problematic_geoloc[subnet][(target_m, target_n)] = [
                        (vp_i_lat, vp_i_lon, min_rtt_j),
                        (vp_j_lat, vp_j_lon, min_rtt_j),
                    ]

    return problematic_geoloc


def get_latencies_per_vp(representatives: list) -> dict[list]:
    """return a dictionnary with each VPs shortest ping to 3 representative targets"""
    latencies_per_vp = defaultdict(list)
    for _, pings in representatives:
        for _, vp_id, min_rtt in pings:
            latencies_per_vp[vp_id].append(min_rtt)

    return latencies_per_vp


def get_reduced_vps(representatives: list, nb_vps: int = 10) -> list[str]:
    """get the 10 VPs with the lowest median latency to the representatives IP addrs"""
    latencies_per_vp = get_latencies_per_vp(representatives)

    # get median rtt per VP
    median_latencies_per_vp = []
    for vp_id, rtts in latencies_per_vp.items():
        median_latencies_per_vp.append((vp_id, median(rtts)))

    # order by median RTT
    median_latencies_per_vp = sorted(median_latencies_per_vp, key=lambda x: x[-1])

    # take the N VPs with the lowest median latency
    reduced_vps = [vp_id for vp_id, _ in median_latencies_per_vp[:nb_vps]]

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
            for vp_addr, vp_id, min_rtt in pings:
                if vp_id not in reduced_vps:
                    continue

                reduced_pings.append((vp_addr, vp_id, min_rtt))

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
    we ought to evaluate three things:
        1. The fraction of IP addresses we geolocate within the same area within each /24
        (using latency and elected VP)
        2. Evaluate the same metric but for each AS type (as we did for continental geoloc)
        3. Re-evaluate the results described in the Million scale paper
    """
    do_fraction_subnet_eval: bool = False
    do_reduced_eval: bool = False
    do_per_asn_eval: bool = True
    do_find_non_overlapping: bool = False

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

        logger.info(f"Under 2ms geoRes  :: {georesolver_under_2_ms}")
        logger.info(f"Under 2ms reduced :: {reduced_under_2_ms}")

    if do_per_asn_eval:

        ordered_curves = []
        frac_per_subnet = get_frac_per_subnet(pings_per_subnet, 2)
        x, y = ecdf(frac_per_subnet)
        frac_under_100 = get_proportion_over(x, y, 1)
        logger.info(f"Fraction all under 2ms, all ASes:: {frac_under_100}")
        ordered_curves.append((frac_under_100, x, y, "All ASes"))

        frac_per_subnet_per_AS_type, nb_subnets, nb_asn = (
            get_frac_per_subnet_per_asn_type(
                pings_per_subnet=pings_per_subnet, latency_threshold=2
            )
        )

        for as_type, frac_per_subnet in frac_per_subnet_per_AS_type.items():
            x, y = ecdf(frac_per_subnet)
            frac_under_100 = get_proportion_over(x, y, 1)
            logger.info(f"Fraction all under 2ms, {as_type}:: {frac_under_100}")
            fraction_of_subnets = round((len(frac_per_subnet) / nb_subnets) * 100, 2)
            ordered_curves.append((frac_under_100, x, y, f"{as_type}"))

        ordered_curves = sorted(ordered_curves, key=lambda x: x[0], reverse=True)
        cdfs = []
        for _, x, y, label in ordered_curves:
            cdfs.append((x, y, label))

        plot_multiple_cdf(
            cdfs=cdfs,
            output_path="subnet_aggregation__fraction_under_2ms_per_as_type",
            metric_evaluated="",
            x_limit_left=0,
            y_label="CDF of subnets",
            x_label="Fraction of IP addresses in the same metro",
            x_log_scale=False,
            legend_size=8,
            legend_fontsize=6,
        )

    if do_find_non_overlapping:
        vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
        # vps_per_id = get_vp_per_id(vps)
        asndb = pyasn(str(path_settings.RIB_TABLE))
        _, vps_coordinates = get_parsed_vps(vps, asndb)

        problematic_geoloc = non_overlapping_geoloc(pings_per_subnet, vps_coordinates)
        logger.info(f"Nb subnet with conflicting geoloc:: {len(problematic_geoloc)}")
        dump_pickle(problematic_geoloc, RESULT_PATH / "problematic_geoloc.pickle")


if __name__ == "__main__":
    main()
