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
from geogiant.common.geoloc import distance
from geogiant.common.files_utils import dump_pickle
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def ref_shortest_ping_eval(
    targets: list,
    ping_vps_to_target: dict,
    vps_coordinates: dict,
) -> tuple[dict, dict]:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    results = {}
    for target in tqdm(targets):

        target = parse_target(target, asndb)

        try:
            pings = ping_vps_to_target[target["addr"]]
            ref_shortest_ping_addr, ref_min_rtt = min(pings, key=lambda x: x[-1])
        except KeyError:
            logger.debug(f"No ping available for target:: {target['addr']}")
            continue

        ref_shortest_ping_vp = get_vp_info(
            target,
            None,
            ref_shortest_ping_addr,
            vps_coordinates,
            ref_min_rtt,
        )

        if not ref_shortest_ping_vp:
            continue

        results[target["addr"]] = {
            "target": target,
            "ref_shortest_ping_vp": ref_shortest_ping_vp,
        }

    return results


def main() -> None:
    asndb = pyasn(str(path_settings.RIB_TABLE))

    removed_vps = path_settings.DATASET / "removed_vps.json"

    ping_vps_to_target = get_pings_per_target(clickhouse_settings.PING_VPS_TO_TARGET)

    targets = load_targets(clickhouse_settings.VPS_FILTERED)
    vps = load_vps(clickhouse_settings.VPS_FILTERED)

    _, vps_coordinates = get_parsed_vps(vps, asndb)

    logger.info("Reference shortest ping evaluation")

    ref_shortest_ping_results = ref_shortest_ping_eval(
        targets=targets,
        ping_vps_to_target=ping_vps_to_target,
        vps_coordinates=vps_coordinates,
    )

    dump_pickle(
        data=ref_shortest_ping_results,
        output_file=path_settings.RESULTS_PATH / f"results_ref_shortest_ping.pickle",
    )


if __name__ == "__main__":
    main()
