"""methods for plotting graph"""

import numpy as np


from pathlib import Path
from loguru import logger
from collections import defaultdict, OrderedDict

from geogiant.common.files_utils import load_pickle, load_countries_continent
from geogiant.common.utils import EvalResults
from geogiant.clickhouse.queries import load_targets
from geogiant.evaluation.evaluation_plot_functions import ecdf, get_proportion_under, plot_multiple_cdf
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def ecs_sp_per_continent(results: dict, metric_evaluated: str) -> None:
    cdfs = []
    all_countries = set()
    geoloc_error_per_continent = defaultdict(list)
    continent_per_country = load_countries_continent()

    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    parsed_targets = {}
    for target in targets:
        parsed_targets[target["addr"]] = target

    for target_addr, target_results_per_metric in results.items():
        target = parsed_targets[target_addr]
        target_continent = continent_per_country[target["country_code"]]

        all_countries.add(target["country_code"])

        try:
            ecs_shortest_ping_vp = target_results_per_metric["result_per_metric"][
                "jaccard"
            ]["ecs_shortest_ping_vp_per_budget"][50]
        except KeyError:
            logger.debug(f"No results for subnet:: {target_addr}")
            continue

        if ecs_shortest_ping_vp[metric_evaluated]:
            geoloc_error_per_continent[target_continent].append(
                ecs_shortest_ping_vp[metric_evaluated]
            )

    for continent, results in geoloc_error_per_continent.items():

        x, y = ecdf(results)
        label = f"{continent} ({len(results)} IP addresses)"
        cdfs.append((x, y, label))

        m_error = round(np.median(x), 2)
        proportion_of_ip = get_proportion_under(x, y)
        logger.info(f"ECS SP:: {continent}, <40km={round(proportion_of_ip, 2)}")
        logger.info(f"ECS SP:: {continent}, median_error={round(m_error, 2)} [km]")

    return cdfs


def main() -> None:
    eval: EvalResults = load_pickle(
        path_settings.RESULTS_PATH
        / "tier4_evaluation/results__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    )

    ecs_cdfs = ecs_sp_per_continent(eval.results_answer_subnets, "d_error")

    plot_multiple_cdf(
        ecs_cdfs,
        output_path="geoloc_error_per_continent",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


if __name__ == "__main__":
    main()
