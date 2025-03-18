# idea: if all VPs in the same country, then IP address in country

# methodo: eval on RIPE Altas anchors

# 2. load targets score and select N first VPs with highest similarity
# 3. compute main country (most represented country in VPs dataset)
# 4. compute target country geoloc vs. main country VP proportion

from pyasn import pyasn
from loguru import logger
from ipaddress import IPv4Network

from georesolver.clickhouse.queries import load_vps, load_targets, get_pings_per_target
from georesolver.common.files_utils import load_pickle, load_json
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.utils import get_parsed_vps, TargetScores
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_vps_country_proportion(
    target_subnet: str,
    target_score: list,
    vps_per_subnet: dict,
    vps_coordinates: dict,
    nb_vps_threshold: int = 50,
) -> dict:

    vps_countries = []

    # for each vps, get country
    try:
        target_score = target_score["jaccard"]
    except TypeError:
        pass

    for vp_subnet, _ in target_score[:nb_vps_threshold]:
        if target_subnet == vp_subnet:
            continue

        vps_in_subnet = vps_per_subnet[vp_subnet]

        for vp_addr in vps_in_subnet:
            _, _, vp_country_code, _ = vps_coordinates[vp_addr]
            vps_countries.append(vp_country_code)

    # get proportion of each represented country
    country_proportions = []
    country_set = set(vps_countries)
    for country in country_set:
        country_proportions.append(
            (country, vps_countries.count(country) / len(vps_countries))
        )

    # sort per most represented country
    country_proportions = sorted(country_proportions, key=lambda x: x[-1], reverse=True)

    return country_proportions


def country_geoloc_fct_proportion(
    targets: list,
    target_scores: dict,
    vps_per_subnet: dict,
    vps_coordinates: dict,
    thresholds_results: list = [0.8, 0.9, 1],
) -> dict:

    results_fct_proportion = {}
    for threshold in thresholds_results:
        correct_country_geoloc = 0
        above_threshold = 0
        target_evaluated = 0
        for target in targets:
            target_subnet = target["subnet"]
            target_country = target["country_code"]
            target_score = target_scores[target_subnet]

            vps_country_proportion = get_vps_country_proportion(
                target_subnet=target_subnet,
                target_score=target_score,
                vps_per_subnet=vps_per_subnet,
                vps_coordinates=vps_coordinates,
                nb_vps_threshold=50,
            )

            if not vps_country_proportion:
                continue

            main_country = vps_country_proportion[0][0]

            if vps_country_proportion[0][1] >= threshold:
                above_threshold += 1

            if vps_country_proportion[0][1] < threshold:
                continue

            target_evaluated += 1

            if main_country == target_country:
                correct_country_geoloc += 1

        fraction_above_threshold = correct_country_geoloc / target_evaluated
        fraction_total = correct_country_geoloc / len(targets)

        results_fct_proportion[threshold] = (
            fraction_above_threshold,
            fraction_total,
            above_threshold,
        )

    return results_fct_proportion


def ripe_atlas_dataset_eval() -> None:
    asndb = pyasn(str(path_settings.RIB_TABLE))
    thresholds_results = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # 1. load targets and VPs
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    # 2. load target scores
    target_scores: TargetScores = load_pickle(
        path_settings.RESULTS_PATH
        / "tier4_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    )
    target_scores = target_scores.score_answer_subnets

    results_fct_proportion = country_geoloc_fct_proportion(
        targets=targets,
        target_scores=target_scores,
        vps_per_subnet=vps_per_subnet,
        vps_coordinates=vps_coordinates,
        thresholds_results=thresholds_results,
    )

    for proportion, results in results_fct_proportion.items():
        logger.info(f"Main country representation:: {round(proportion * 100, 2)} [%]")
        logger.info(f"Correct geoloc > proportion:: {round(results[0] * 100, 2)} [%]")
        logger.info(f"Correct geoloc total:: {round(results[1] * 100 , 2)} [%]")
        logger.info(f"Above threshold:: {results[2]}\n")


def geoLite_dataset_eval() -> None:
    asndb = pyasn(str(path_settings.RIB_TABLE))
    geoLite_db = load_json(
        path_settings.DATASET
        / "GeoLite2-Country-CSV_20240702/geoLite_subnets_countries.json"
    )

    # 1. load targets under 2 ms and there best VPs
    pings_per_target = get_pings_per_target("pings_internet_scale")
    parsed_targets = []
    target_scores = {}

    for i, (target, ping_vps) in enumerate(pings_per_target.items()):
        target_subnet = get_prefix_from_ip(target)

        try:
            target_country = geoLite_db[target_subnet]
        except KeyError:
            continue

        parsed_targets.append(
            {
                "subnet": target_subnet,
                "country_code": target_country,
            }
        )

        target_scores[target_subnet] = list(
            set([(get_prefix_from_ip(vp_addr), ping) for vp_addr, ping in ping_vps])
        )

    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)

    results_fct_proportion = country_geoloc_fct_proportion(
        targets=parsed_targets,
        target_scores=target_scores,
        vps_per_subnet=vps_per_subnet,
        vps_coordinates=vps_coordinates,
        thresholds_results=[0.5, 0.6, 0.7, 0.8, 0.9],
    )

    for proportion, results in results_fct_proportion.items():
        logger.info(f"Main country representation:: {round(proportion * 100, 2)} [%]")
        logger.info(f"Correct geoloc > proportion:: {round(results[0] * 100, 2)} [%]")
        logger.info(f"Correct geoloc total:: {round(results[1] * 100 , 2)} [%]")
        logger.info(f"Above threshold:: {results[2]}")
        logger.info(f"Total targets analyzed:: {len(parsed_targets)}\n")


def ipInfo_dataset_eval() -> None:
    pass
    # TODO


if __name__ == "__main__":
    ripe_atlas_dataset_eval()
