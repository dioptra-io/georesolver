"""plot results for subnet aggregation experiments"""

from random import sample
from collections import defaultdict

from georesolver.clickhouse.queries import load_vps, get_pings_per_target
from georesolver.evaluation.evaluation_plot_functions import ecdf, get_proportion_under
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


def get_proportion_per_subnet(pings_per_subnet: dict[list]) -> list:
    """calculate the fraction of IP addresses with a latency under 2ms"""
    under_2_ms_per_subnet = []
    for _, pings_per_target in pings_per_subnet.items():
        shortest_pings_per_ip = []

        # get the shortest ping per target
        for _, pings in pings_per_target:
            shortest_ping = min([ping[-1] for ping in pings])
            shortest_pings_per_ip.append(shortest_ping)

        # get the proportion under 2 ms
        x, y = ecdf(shortest_pings_per_ip)
        under_2_ms = get_proportion_under(x, y, threshold=2)

        under_2_ms_per_subnet.append(under_2_ms)

    return under_2_ms_per_subnet


def get_subnet_vps(representatives: list) -> dict:
    """get the 10 VPs with the lowest median latency to the representatives IP addrs"""
    latencies_per_vp = defaultdict(list)
    for _, pings in representatives:
        for ping in pings:
            pass  # TODO


def reduced_georesolver_evaluation(pings_per_subnet: dict[list]) -> list:
    """
    evaluate the million scale paper results:
        - take three random representative IP addresses per subnet
        - select the 10 VPs with the lowest median error
        - get the results for the rest of the IP addresses
        - compare the results
    """
    for subnet, pings_per_target in pings_per_subnet.items():
        if not len(pings_per_target) > 3:
            continue

        # extract three random targets
        representatives = sample(pings_per_target, 3)

        # get median latency per vps
        subnet_vps = get_subnet_vps()


def main() -> None:
    """
    we outght to evaluate three thinks:
        1. The fraction of IP addresses we geolocate within the same area within each /24
        (using latency and elected VP)
        2. Evaluate the same metric but for each AS type (as we did for continental geoloc)
        3. Re-evaluate the results described in the Million scale paper
    """
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)

    pings_per_subnet = load_pings_per_subnet()
    under_2_ms_per_subnet = get_proportion_per_subnet(pings_per_subnet)

    # TODO: get ecdf and plot


if __name__ == "__main__":
    main()
