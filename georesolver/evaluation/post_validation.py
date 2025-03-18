"""based on georesolver measurement, check for SOI violation and itteratively remove VPs with the most SOI"""

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from collections import defaultdict

from georesolver.clickhouse.queries import iter_pings_per_target, load_vps, get_targets
from georesolver.common.geoloc import distance, get_max_theoretical_dst
from georesolver.common.files_utils import load_json, dump_json
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

VPS_TABLE = "vps_filtered_final_CoNEXT_winter_submision"
PING_TABLE = "itdk_ping"


def get_soi(
    pings: list[tuple],
    vps_per_id: dict[dict],
    soi_per_id: dict[set],
    soi_dst_threshold: int = 50,
) -> None:
    """for a set of pings, return vps with SOI"""
    for _, vp_i_id, min_rtt_i in pings:
        vp_i = vps_per_id[vp_i_id]
        vp_i_lat, vp_i_lon = vp_i["lat"], vp_i["lon"]

        # get max theoretical dst between vp_i and target
        vp_i_dst = get_max_theoretical_dst(min_rtt_i)

        for _, vp_j_id, min_rtt_j in pings:
            if vp_i_id == vp_j_id:
                continue

            vp_j = vps_per_id[vp_j_id]
            vp_j_lat, vp_j_lon = vp_j["lat"], vp_j["lon"]

            # get max theoretical dst between vp_j and target
            vp_j_dst = get_max_theoretical_dst(min_rtt_j)

            # get dst between vps
            inter_vps_dst = distance(vp_i_lat, vp_j_lat, vp_i_lon, vp_j_lon)

            # avoid adding a SOI if the soi dst is too small
            if inter_vps_dst > soi_dst_threshold:
                continue

            # check for speed of light violation between the two measurements
            if (vp_i_dst + vp_j_dst) < inter_vps_dst:
                # Impossible distance
                soi_per_id[vp_i_id].add(vp_j_id)
                soi_per_id[vp_j_id].add(vp_i_id)

    return soi_per_id


def get_removed_vps_from_itdk_pings(
    vps_per_id: dict[dict], output_path: Path, soi_dst_threhsold: int = 50
) -> list[tuple]:
    """compute wrongly geolocated vps based on greedy SOI removal"""

    if not output_path.exists():

        # for logging with tqdm
        targets = get_targets(PING_TABLE)
        nb_rows = len(targets)

        logger.info(f"Retrieved pings for {len(targets)}")
        rows = iter_pings_per_target(PING_TABLE)

        # get SOI based on GeoResolver measurements
        soi_per_id = defaultdict(set)
        for row in tqdm(rows, total=nb_rows):
            pings = row["pings"]

            soi_per_id = get_soi(pings, vps_per_id, soi_per_id, soi_dst_threhsold)

        # itteratively remove vps until we no longer have SOI
        # Greedily remove the vp_id with the more SOI violations
        removed_vps = set()
        n_violations = sum([len(x) for x in soi_per_id.values()])
        while n_violations > 0:
            logger.info(f"Violations: {n_violations}")
            # Remove the vp id with the highest number of SOI violations
            worse_vp_id, vps_soi = max(soi_per_id.items(), key=lambda x: len(x[1]))

            # remove SOI previously associated with worse vp
            for vp_id, vps_soi in soi_per_id.items():
                vps_soi.discard(worse_vp_id)

            # del worse vp from soi and save
            del soi_per_id[worse_vp_id]
            worse_vp = vps_per_id[worse_vp_id]
            removed_vps.add((worse_vp_id, worse_vp["addr"]))

            # recalculate total soi violations
            n_violations = sum([len(x) for x in soi_per_id.values()])

        logger.info(f"{len(removed_vps)=}")

        # filter measurement based on removed vps
        dump_json(list(removed_vps), output_path)

        return removed_vps

    removed_vps = load_json(output_path)
    return removed_vps


def filter_results(
    vps_per_id: dict[dict], removed_vps: list[tuple], dst_threshold: int = 50
) -> None:
    """filter ping results based on found SOI"""
    targets = get_targets(PING_TABLE, threshold=2)
    nb_rows = len(targets)

    logger.info(f"Metro precision targets before filtering:: {len(targets)}")

    removed_vps_ids = set([id for id, _ in removed_vps])

    rows = iter_pings_per_target(PING_TABLE, threshold=2)

    # removed all filtered vps from ping results
    major_soi_vp = set()
    soi_dst_per_target = defaultdict(list)
    filtered_pings_per_target = defaultdict(list)
    filtered_pings_per_target_strict = defaultdict(list)
    for row in tqdm(rows, total=nb_rows):
        target = row["target"]
        pings = row["pings"]

        # find the shortest ping vp not within removed vps
        shortest_ping_vp = None
        pings = sorted(pings, key=lambda x: x[-1], reverse=True)
        for shortest_ping in pings:
            if shortest_ping[1] in removed_vps_ids:
                continue

            shortest_ping_vp = vps_per_id[shortest_ping[1]]

        if not shortest_ping_vp:
            logger.error(f"Cannot find a vp without SOI for {target=}")
            continue

        for vp_addr, vp_id, min_rtt in pings:
            # filter vp result based on removed vp list
            if vp_id in removed_vps_ids:
                # get missing dst between shortest ping and removed vp
                soi_vp = vps_per_id[vp_id]

                soi_dst = distance(
                    shortest_ping_vp["lat"],
                    soi_vp["lat"],
                    shortest_ping_vp["lon"],
                    soi_vp["lon"],
                )

                if soi_dst > dst_threshold:
                    # save soi dst with target info
                    major_soi_vp.add((vp_id, vp_addr))
                    soi_dst_per_target[target].append(
                        {
                            "shortest_ping_vp_addr": shortest_ping[0],
                            "shortest_ping_vp_id": shortest_ping[1],
                            "shortest_ping_rtt": shortest_ping[2],
                            "soi_vp_addr": vp_addr,
                            "soi_vp_id": vp_id,
                            "soi_vp_rtt": min_rtt,
                            "soi_dst": soi_dst,
                        }
                    )

                # on small distances, SOI is too strict
                if soi_dst < dst_threshold:
                    filtered_pings_per_target[target].append((vp_addr, vp_id, min_rtt))
                    continue

            filtered_pings_per_target[target].append((vp_addr, vp_id, min_rtt))
            filtered_pings_per_target_strict[target].append((vp_addr, vp_id, min_rtt))

    # save data
    dump_json(soi_dst_per_target, path_settings.DATASET / "soi_dst_per_target.json")
    dump_json(list(major_soi_vp), path_settings.DATASET / "major_soi_vp.json")

    # get shortest ping and plot results
    metro_geoloc = {}
    for target, pings in filtered_pings_per_target.items():
        # get shortest ping
        shortest_ping = min(pings, key=lambda x: x[-1])

        # only keep targets with metropolitan precision
        if not shortest_ping[-1] < 2:
            continue

        metro_geoloc[target] = shortest_ping

    logger.info(f"Metro precision targets after filtering:: {len(metro_geoloc)}")
    logger.info(f"Number of major SOI vps :: {len(major_soi_vp)}")


def main() -> None:
    """entry point"""
    vps = load_vps(VPS_TABLE)
    vps_per_id = {}
    for vp in vps:
        vps_per_id[vp["id"]] = vp

    # get new metrics (the number of IP addresses under 2ms)
    removed_vps = get_removed_vps_from_itdk_pings(
        vps_per_id, path_settings.DATASET / "itdk_removed_vps_with_dst_threshold.json"
    )

    logger.info(
        f"Number of removed VPs based on pings measurements:: {len(removed_vps)}"
    )

    filter_results(vps_per_id, removed_vps)


if __name__ == "__main__":
    main()
