"""load pings from clickhouse"""

import os

from georesolver.clickhouse.queries import load_vps, load_targets, get_pings_per_target
from georesolver.common.files_utils import load_json
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
os.environ["CLICKHOUSE_DATABASE"] = ch_settings.CLICKHOUSE_DATABASE_EVAL

VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
MESHED_PING_TABLE = "vps_meshed_pings_deprecated"
ROUTER_DB = "pings_end_to_end_true_ids"
ROUTER_TARGETS = path_settings.USER_DATASETS / "router_targets.json"
REMOVED_VPS = path_settings.USER_DATASETS / "removed_vps.json"


def main() -> None:
    """entry point"""
    # RIPE Atlas anchors
    targets = load_targets(VPS_TABLE)
    # RIPE Atlas anchors + vps
    vps = load_vps(VPS_TABLE)
    # meshed pings contain wrong vps
    removed_vps = load_json(REMOVED_VPS)

    pings_per_target = get_pings_per_target(MESHED_PING_TABLE, removed_vps)

    for target, pings in pings_per_target.items():
        print(f"{target=}; {len(pings)=}")
        for vp_addr, rtts in pings:
            print(f"{vp_addr=}; {rtts=}")
            break

        break


if __name__ == "__main__":
    main()
