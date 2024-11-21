""""evaluation of georesolver on itdk dataset, include comparison and evaluation of hoiho geolocatio"""

from loguru import logger

from georesolver.clickhouse.queries import get_pings_per_target
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_cdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.common.files_utils import load_csv, load_json, dump_csv, dump_json
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

GEORESOLVER_ITDK_PING_TABLE = "itdk_ping"


def get_georesolver_shortest_ping() -> dict[float]:
    """retrieve georesolver shortest ping"""
    pings_per_target = get_pings_per_target(GEORESOLVER_ITDK_PING_TABLE)

    shortest_ping_per_target = {}
    for target, pings in pings_per_target.items():
        _, min_rtt = min(pings, key=lambda x: x[-1])

        shortest_ping_per_target[target] = min_rtt

    return shortest_ping_per_target


def georesolver_evaluation() -> None:
    """simple georesolver evaluation on ITDK dataset, output CDF of the latency"""
    shortest_ping_per_target = get_georesolver_shortest_ping()
    x, y = ecdf(shortest_ping_per_target.values())
    frac_under_2_ms = get_proportion_under(x, y, threshold=2)

    logger.info(
        f"Fraction of IP addresses under 2 ms:: {round(frac_under_2_ms * 100,2)}[%]"
    )

    plot_cdf(
        x=x,
        y=y,
        output_path="georesolver_itdk_cdf",
        x_label="RTT (ms)",
        y_label="CDF of targets",
        metric_evaluated="rtt",
    )


def coverage_evaluation() -> None:
    """
    evaluate coverage of georesolver on ITDK dataset:
        - fraction of responsive IP address
        - fraction of IP addresses under 2ms
        - fraction of Hoiho geolocation
        - fraction of intersection
        - fraction of Hoiho IP address for which we do not have geolocation
        - fraction of Georesolver IP address for which Hoiho does not have geolocation
    """
    pass


def main() -> None:
    """
    entry point of itdk/hoiho evaluation:
        - raw evaluation of georesolver on ITDK dataset:
            - fraction/cdf of georesolver latency
        - coverage evaluation:
            - fraction of IP addresses of:
                - hoiho intersection georesolver
                - hoiho IP addresses we do not geolocate
                - ITDK IP addresses georesolver geolocate but not hoiho
        - hoiho verification:
            - IP addresses for which we confirm the geolocation
            - fraction of IP addresses with invalid geoloction in hoiho dataset
    """
    do_georesolver_evaluation: bool = True

    if do_georesolver_evaluation:
        georesolver_evaluation()


if __name__ == "__main__":
    main()
