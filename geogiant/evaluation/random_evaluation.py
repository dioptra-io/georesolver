from random import sample
from pyasn import pyasn
from tqdm import tqdm
from loguru import logger

from geogiant.common.queries import (
    get_pings_per_target,
    load_targets,
    load_vps,
)
from geogiant.common.utils import (
    get_parsed_vps,
    parse_target,
    get_vp_info,
)
from geogiant.common.utils import TargetScores
from geogiant.common.files_utils import load_json, dump_pickle, load_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def random_shortest_ping_eval(
    targets: list,
    ping_vps_to_target: dict,
    vps_coordinates: dict,
    subnet_scores: dict = None,
) -> tuple[dict, dict]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    for target in tqdm(targets):

        target = parse_target(target, asndb)

        if subnet_scores:
            try:
                target_scores: dict = subnet_scores[target["subnet"]]
            except KeyError:
                logger.error(
                    f"cannot find target score for subnet : {target['subnet']}"
                )
                continue

        try:
            pings = ping_vps_to_target[target["addr"]]
            pings = sample(pings, 50)
            random_shortest_ping_addr, random_min_rtt = min(pings, key=lambda x: x[-1])
        except KeyError:
            logger.debug(f"No ping available for target:: {target['addr']}")
            continue

        random_shortest_ping_vp = get_vp_info(
            target,
            target_scores["jaccard"],
            random_shortest_ping_addr,
            vps_coordinates,
            random_min_rtt,
        )

        if not random_shortest_ping_vp:
            continue

        results[target["addr"]] = {
            "target": target,
            "random_shortest_ping_vp": random_shortest_ping_vp,
        }

    return results


def main() -> None:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    removed_vps = load_json(path_settings.REMOVED_VPS)
    ping_vps_to_target = get_pings_per_target(
        clickhouse_settings.VPS_VPS_MESHED_PINGS_TABLE, removed_vps
    )

    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)

    _, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps=removed_vps)

    logger.info("Random shortest ping evaluation")

    score_file: TargetScores = load_pickle(
        path_settings.RESULTS_PATH
        / "tier4_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    )

    random_shortest_ping_results = random_shortest_ping_eval(
        targets=targets,
        subnet_scores=score_file.score_answer_subnets,
        ping_vps_to_target=ping_vps_to_target,
        vps_coordinates=vps_coordinates,
    )

    dump_pickle(
        data=random_shortest_ping_results,
        output_file=path_settings.RESULTS_PATH / f"results_random_shortest_ping.pickle",
    )


if __name__ == "__main__":
    main()
