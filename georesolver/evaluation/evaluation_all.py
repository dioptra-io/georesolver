"""central script for running georesolver evaluation CoNEXT 2025 evaluation"""

from loguru import logger

from georesolver.evaluation import (
    figure_2_center_figure_5_right,
    figure_2_left_right,
    figure_3_all,
    figure_4_left,
    figure_4_right,
    figure_5_left_center,
    table_1,
    georesolver_vs_hoiho,
    georesolver_vs_single_radius,
    itdk_post_validation,
    local_resolver_vs_gpdns,
    measurement_overhead,
    subnet_aggregation,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


def main(
    do_figure_2_center_figure_5_right: bool = False,
    do_figure_2_left_right: bool = False,
    do_figure_3_all: bool = False,
    do_figure_4_left: bool = False,
    do_figure_4_right: bool = False,
    do_figure_5_left_center: bool = False,
    do_table_1: bool = False,
    do_georesolver_vs_hoiho: bool = False,
    do_georesolver_vs_single_radius: bool = False,
    do_itdk_post_validation: bool = False,
    do_local_resolver_vs_gpdns: bool = False,
    do_measurement_overhead: bool = True,
    # other scripts, not present in GeoResolver's paper
    do_subnet_aggregation: bool = False,
) -> None:
    """entry point, set booleans to True or False to either run evaluation or not"""
    if do_figure_2_center_figure_5_right:
        logger.info("** Figure 2 center: CDNs IP addrs and Figure 5 right:  **")
        figure_2_center_figure_5_right.main(
            do_measurement=False,
            do_plot=True,
            do_additional_plot=False,
        )

    if do_figure_2_left_right:
        logger.info(
            "** Figure 2 left: RIPE Atlas Anchors and right: Latency threshold **"
        )
        figure_2_left_right.main(
            do_compute_score=True,
            do_evaluation=True,
            do_plot=True,
        )

    if do_figure_3_all:
        logger.info("** Figure 3: Design Decision **")
        figure_3_all.main(
            do_compute_score=True,
            do_evaluation=True,
            do_plot=True,
        )

    if do_figure_4_left:
        logger.info("** Figure 4 left: HyperGiants dependency **")
        figure_4_left.main(
            do_load_hostnames=True,
            do_compute_score=True,
            do_evaluation=True,
            do_plot=True,
        )

    if do_figure_4_right:
        logger.info("** Figure 4 right: Private DB comparison **")
        figure_4_right.main(
            do_maxmind_measurements=False,
            do_ip_info_measurements=False,
            do_evaluation=True,
        )

    if do_figure_5_left_center:
        logger.info("** Figure 5 left: Probing budget and center: VPs ranking **")
        figure_5_left_center.main(
            do_compute_score=True,
            do_evaluation=True,
            do_plot=True,
        )

    if do_table_1:
        logger.info("** Table 1: Hostnames selection **")
        table_1.main(
            do_compute_score=False,
            do_evaluation=False,
            do_plot=True,
        )

    if do_georesolver_vs_hoiho:
        logger.info("** GeoResolver vs. Hoiho **")
        georesolver_vs_hoiho.main(
            do_georesolver_evaluation=True,
            do_coverage_evaluation=True,
            do_hoiho_vs_georesolver=True,
            do_projection_evaluation=True,
        )

    if do_georesolver_vs_single_radius:
        logger.info("** GeoResolver vs. Single Radius **")
        georesolver_vs_single_radius.main(
            do_load_dataset=False,
            do_measurements=False,
            do_evaluation=True,
        )

    if do_itdk_post_validation:
        logger.info("** GeoResolver post measurement validation **")
        itdk_post_validation.main()

    if do_local_resolver_vs_gpdns:
        logger.info("** Local resolver vs. GPDNS **")
        local_resolver_vs_gpdns.main()

    if do_measurement_overhead:
        logger.info("** Measurement overhead **")
        measurement_overhead.main()


if __name__ == "__main__":
    main()
