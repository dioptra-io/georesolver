""""evaluation of georesolver on itdk dataset, include comparison and evaluation of hoiho geolocatio"""

from loguru import logger
from tqdm import tqdm
from pyasn import pyasn
from matplotlib_venn import venn3
from matplotlib import pyplot as plt

from georesolver.clickhouse.queries import get_pings_per_target, load_vps
from georesolver.evaluation.evaluation_plot_functions import ecdf, plot_cdf
from georesolver.common.utils import get_parsed_vps
from georesolver.common.geoloc import is_within_cirle
from georesolver.common.files_utils import load_csv, load_json, dump_csv, dump_json
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

GEORESOLVER_ITDK_PING_TABLE = "itdk_ping"
GEORESOLVER_PING_RESULTS = path_settings.RESULTS_PATH / "georesolver_ping_results.json"
GEORESOLVER_REMOVED_TARGETS = (
    path_settings.RESULTS_PATH / "georesolver_removed_targets.csv"
)


def get_georesolver_shortest_ping() -> tuple[dict]:
    """retrieve georesolver shortest ping"""
    initial_targets = load_csv(
        path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv"
    )
    pings_per_target = get_pings_per_target(GEORESOLVER_ITDK_PING_TABLE)

    shortest_ping_per_target = {}
    under_2_ms = {}
    for target in tqdm(initial_targets):
        try:
            pings = pings_per_target[target]
        except KeyError:
            continue

        vp, min_rtt = min(pings, key=lambda x: x[-1])

        shortest_ping_per_target[target] = (vp, min_rtt)

        if min_rtt <= 2:
            under_2_ms[target] = (vp, min_rtt)

    return shortest_ping_per_target, under_2_ms


def georesolver_evaluation() -> None:
    """simple georesolver evaluation on ITDK dataset, output CDF of the latency"""
    shortest_ping_per_target, under_2_ms = get_georesolver_shortest_ping()
    frac_under_2_ms = len(under_2_ms) / len(shortest_ping_per_target) * 100

    logger.info(f"Nb targets geolocated :: {len(shortest_ping_per_target)}")
    logger.info(f"Nb targets under 2 ms :: {len(under_2_ms)}")
    logger.info(f"Fraction under 2 ms   :: {round(frac_under_2_ms,2)}[%]")

    x, y = ecdf([min_rtt for _, min_rtt in shortest_ping_per_target.values()])
    plot_cdf(
        x=x,
        y=y,
        x_log_scale=True,
        x_lim_right=1_00,
        output_path="georesolver_itdk_cdf",
        x_label="RTT (ms)",
        y_label="CDF of targets",
        metric_evaluated="rtt",
    )


def coverage_evaluation() -> None:
    """
    evaluate coverage of georesolver on ITDK dataset:
        - fraction of responsive IP address
    """
    all_router_interfaces = load_csv(path_settings.DATASET / "itdk/itdk_addrs_all.csv")
    resp_router_interfaces = load_csv(
        path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv"
    )
    geoloc_hoiho = load_json(path_settings.DATASET / "itdk/hoiho_parsed_geoloc.json")
    hoiho_targets = [t for t in geoloc_hoiho]
    responsive_hoiho_targets = set(resp_router_interfaces).intersection(
        set(hoiho_targets)
    )
    hoiho_itdk_coverage = round(
        (len(hoiho_targets) / len(all_router_interfaces)) * 100, 2
    )
    frac_responsive_hoiho = round(
        (len(responsive_hoiho_targets) / len(geoloc_hoiho)) * 100, 2
    )
    frac_georesolver = round((2_030_192 / len(all_router_interfaces)) * 100, 2)

    logger.info("HOIHO coverage             ::")
    logger.info(f"Router iffaces            :: {len(hoiho_targets)}")
    logger.info(f"Coverage over ITDK router interfaces:: {hoiho_itdk_coverage}")
    logger.info(f"Responsive router iffaces :: {len(responsive_hoiho_targets)} ")
    logger.info(f"Responsive router iffaces :: {frac_responsive_hoiho}[%]")
    logger.info("GEORESOLVER coverage       ::")
    logger.info(f"Router interfaces (responsive) :: 2.03M ")
    logger.info(f"Router interfaces :: {frac_georesolver} [%]")
    logger.info(f"Theoric coverage goeresolver (20%) :: {frac_georesolver} [%]")


def hoiho_geoloc_vs_georesolver() -> None:
    """
    compare Hoiho geolocation against georesolver:
        - fraction of Hoiho geolocation
        - fraction of intersection
        - fraction of Hoiho IP address for which we do not have geolocation
        - fraction of Georesolver IP address for which Hoiho does not have geolocation
    """
    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    _, vps_coordinates = get_parsed_vps(vps, asndb)
    hoiho_geolocation = load_json(
        path_settings.DATASET / "itdk/hoiho_parsed_geoloc.json"
    )
    shortest_ping_per_target, under_2_ms = get_georesolver_shortest_ping()

    hoiho_targets = [t for t in hoiho_geolocation.keys()]
    georesolver_targets = [t for t in shortest_ping_per_target.keys()]

    # simple coverage analysis
    intersection = set(hoiho_targets).intersection(set(georesolver_targets))
    intersection_2_ms = set(hoiho_targets).intersection(set(under_2_ms))
    missing_in_hoiho = set(georesolver_targets).difference(set(hoiho_targets))
    missing_in_hoiho_2_ms = set(under_2_ms).difference(set(hoiho_targets))
    missing_in_georesolver = set(hoiho_targets).difference(set(georesolver_targets))

    logger.info(f"Intersection size                 :: {len(intersection)}")
    logger.info(f"Intersection size (under 2ms)     :: {len(intersection_2_ms)}")
    logger.info(f"Targets missing hoiho             :: {len(missing_in_hoiho)}")
    logger.info(f"Targets missing hoiho (under 2ms) :: {len(missing_in_hoiho_2_ms)}")
    logger.info(f"Targets missing georesolver       :: {len(missing_in_georesolver)}")

    # check for impossible hoiho geoloc
    imposible_geoloc = {}
    for target in intersection:
        hoiho_country = hoiho_geolocation[target]["country"]
        hoiho_lat = float(hoiho_geolocation[target]["lat"])
        hoiho_lon = float(hoiho_geolocation[target]["lon"])
        vp, min_rtt = shortest_ping_per_target[target]

        try:
            lat, lon, country, _ = vps_coordinates[vp]
        except KeyError:
            # TODO: get VP id that has the shortest ping and find its coordinates
            continue

        is_within = is_within_cirle(
            vp_geo=(lat, lon), rtt=min_rtt, candidate_geo=(hoiho_lat, hoiho_lon)
        )

        if not is_within:
            imposible_geoloc[target] = {
                "hoiho_country": hoiho_country,
                "hoiho_lat": hoiho_lat,
                "hoiho_lon": hoiho_lon,
                "vp": vp,
                "min_rtt": min_rtt,
                "country": country,
                "lat": lat,
                "lon": lon,
            }

    frac_impossible_geoloc = (len(imposible_geoloc) / len(intersection)) * 100

    logger.info(f"Nb impossible geoloc   :: {len(imposible_geoloc)}")
    logger.info(f"Frac impossible geoloc :: {round(frac_impossible_geoloc, 2)}")

    dump_json(
        imposible_geoloc, path_settings.RESULTS_PATH / "hoiho/impossible_geoloc.json"
    )

    return imposible_geoloc


def plot_venn() -> None:
    """plot venn diagramm for ITDK vs. Hoiho vs. georesolver"""
    # Make a Basic Venn
    v = venn3(
        subsets=(1, 1, 1, 1, 1, 1, 1),
        set_labels=("ITDK unresponsive", "ITDK responsive", "Hoiho responsive"),
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"venn_itdk_vs_hoiho_georesolver.png",
        bbox_inches="tight",
    )
    plt.savefig(
        path_settings.FIGURE_PATH / f"venn_itdk_vs_hoiho_georesolver.pdf",
        bbox_inches="tight",
    )


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
    do_coverage_evaluation: bool = True
    do_hoiho_vs_georesolver: bool = True
    do_plot_venn: bool = True

    if do_georesolver_evaluation:
        georesolver_evaluation()

    if do_coverage_evaluation:
        coverage_evaluation()

    if do_hoiho_vs_georesolver:
        impossible_geoloc = hoiho_geoloc_vs_georesolver()

    if do_plot_venn:
        plot_venn()


if __name__ == "__main__":
    main()
