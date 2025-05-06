"""load pings from clickhouse"""
import copy

import maxminddb
import random
import ipaddress
from pathlib import Path

from georesolver.clickhouse.queries import load_vps, load_targets, get_pings_per_target
from georesolver.common.files_utils import load_json, dump_json
from georesolver.common.settings import PathSettings, ClickhouseSettings
from georesolver.common.geoloc import haversine
from georesolver.common.plot import plot_multiple_cdf, homogenize_legend, plot_save
from georesolver.common.ip_addresses_utils import get_prefix_from_ip

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

VPS_TABLE = ch_settings.VPS_FILTERED_FINAL_TABLE
VPS_TABLE_ALL = ch_settings.VPS_ALL_TABLE
MESHED_PING_TABLE = "vps_meshed_pings_CoNEXT_winter_submision"
ROUTER_PING_TABLE = "pings_end_to_end_true_ids"
REMOVED_VPS_CONEXT = path_settings.USER_DATASETS / "removed_vps.json"
REMOVED_VPS_IMC = path_settings.USER_DATASETS / "removed_vps_imc2024.json"
IPINFO_GEOLOCATION_SNAPSHOT = path_settings.USER_DATASETS / "ipinfo_2024-12-01.snapshot"
MM_GEOLOCATION_SNAPSHOT = "/storage/kevin/mm_paid_output_2024-12-03.mmdb"
ROUTER_TARGETS_FILE = path_settings.USER_DATASETS / "router_targets.json"


STRATEGY_CLOSEST_50 = "closest50"
STRATEGY_RANDOM_50_CC = "random_50_cc"
STRATEGY_BRUTE_FORCE = "brute_force"
STRATEGY_DB = "db"
def get_closest_vps(vps, target, lat, lon, n_vps) -> list[str]:


    closest_vps = dict(sorted([(vp["addr"], haversine((vp["lat"], vp["lon"]), (lat, lon)))
                               for vp in vps
                               if vp["addr"] != target
                               and get_prefix_from_ip(vp["addr"]) != get_prefix_from_ip(target)],
                              key=lambda x:x[1])[:n_vps])

    return closest_vps

def get_random_vps_cc(vps, target, cc, n_vps) -> set:

    vps_in_cc = sorted([vp["addr"]
                     for vp in vps
                     if vp ["country_code"] == cc
                        and vp["addr"] != target
                        and get_prefix_from_ip(vp["addr"]) != get_prefix_from_ip(target)])

    vps_in_cc_sample = random.sample(vps_in_cc, min(n_vps, len(vps_in_cc)))

    # Fill with random VPs if we do not have enough probes in the country
    if len(vps_in_cc_sample) < n_vps:
        random_vps = random.sample([vp["addr"] for vp in vps], n_vps - len(vps_in_cc))
        vps_in_cc_sample.extend(random_vps)

    vps_in_cc_sample = set(vps_in_cc_sample)

    return vps_in_cc_sample

def get_geo(geolocation_reader, target, geolocation_database):
    geolocation_response = geolocation_reader.get(target)

    if "ipinfo" in geolocation_database.lower():
        geo_database = "ipinfo"
        lat, lon = float(geolocation_response["lat"]), float(geolocation_response["lng"])
        cc = geolocation_response["country"]
    elif "mm" in geolocation_database.lower():
        geo_database = "maxmind"
        # Strategy 1, get the 50 closest VPs from the geolocation
        if "latitude" not in geolocation_response:
            print(f"{target} not in geolocation database")
            return None
        # location = geolocation_response["location"]
        lat, lon = float(geolocation_response["latitude"]), float(geolocation_response["longitude"])
        cc = geolocation_response["country_iso_code"]
    else:
        raise Exception("No such geo database")

    return lat, lon, cc

def geolocate(targets, pings_per_target, vps, geolocation_reader, geoloc_per_vp, strategy, geolocation_database, table) -> dict:

    # Loads the pings per target

    # Now filter the sources according to different evaluation scenarios
    # Two scenarios
    # one where we select the 50 closest VPs from the commercial database snapshot geolocation of the target
    # one where we select random 50 VPs in the country from the commercial database snapshot geolocation of the target

    assert(strategy in [STRATEGY_CLOSEST_50, STRATEGY_RANDOM_50_CC, STRATEGY_BRUTE_FORCE])
    shortest_ping_per_target = {}

    for target, pings in pings_per_target.items():

        if targets is not None and target not in targets:
            continue

        geolocation_response = get_geo(geolocation_reader, target, geolocation_database)
        if geolocation_response is None:
            continue

        lat, lon, cc = geolocation_response


        if strategy == STRATEGY_CLOSEST_50:
            selected_vps = get_closest_vps(vps, target, lat, lon, n_vps=50)
        elif strategy == STRATEGY_RANDOM_50_CC:
            selected_vps = get_random_vps_cc(vps, target, cc, n_vps=50)
        elif strategy == STRATEGY_BRUTE_FORCE:
            selected_vps = set([vp["addr"] for vp in vps
                                if vp["addr"] != target
                                and get_prefix_from_ip(vp["addr"]) != get_prefix_from_ip(target)])

        min_vp, min_rtt = None, None
        pings_d = dict(pings)
        pings_d_selected = {p: pings_d[p] for p in pings_d if p in selected_vps}
        for vp_addr, rtts in pings_d_selected.items():
            # if vp_addr not in selected_vps:
            #     continue
            if vp_addr not in geoloc_per_vp:
                continue
            # Select VP with the smallest RTT
            min_rtt_vp = rtts
            if min_rtt is None or min_rtt_vp < min_rtt:
                min_rtt = min_rtt_vp
                min_vp = vp_addr

        if len(pings) < 5000 or min_vp is None:
            print(f"{target}; {len(pings)}; min_vp {min_vp} min rtt {min_rtt} {geolocation_database} {strategy} {table}")
        if min_vp is not None:
            shortest_ping_per_target[target] = (min_vp, min_rtt)

    return shortest_ping_per_target


def geolocate_db(targets, geolocation_reader, geolocation_database):

    geoloc_per_target = {}
    for target in targets:
        geolocation_response = get_geo(geolocation_reader, target, geolocation_database)

        if geolocation_response is None:
            continue

        geoloc_per_target[target] = geolocation_response


    return geoloc_per_target






def main() -> None:
    random.seed(42)
    """entry point"""
    # RIPE Atlas anchors
    targets = load_targets(VPS_TABLE)

    anchor_targets = set([t["addr"] for t in targets])
    # RIPE Atlas anchors + vps
    vps = load_vps(VPS_TABLE)
    vps_all = load_vps(VPS_TABLE_ALL)
    vps_per_id ={vp["id"]: vp for vp in vps}
    # removed_vps = set()
    # router targets
    router_targets = set(load_json(Path(ROUTER_TARGETS_FILE)))

    geoloc_per_vp = {vp["addr"] : (vp["lat"], vp["lon"]) for vp in vps_all}
    geoloc_per_vp.update({vp["addr"]: (vp["lat"], vp["lon"]) for vp in targets})


    is_recompute_anchors = True
    is_recompute_routers = True

    for ping_table in [MESHED_PING_TABLE, ROUTER_PING_TABLE]:
        Ys = []
        if ping_table == MESHED_PING_TABLE:
            targets = None
            database = ch_settings.CLICKHOUSE_DATABASE
            # meshed pings contain wrong vps
            removed_vps = load_json(REMOVED_VPS_CONEXT)
        elif ping_table == ROUTER_PING_TABLE:
            targets = router_targets
            database = ch_settings.CLICKHOUSE_DATABASE_EVAL
            removed_vps = load_json(REMOVED_VPS_IMC)

        ofile = f"resources/geolocation_databases_{ping_table}.json"
        if is_recompute_anchors or is_recompute_routers:
            pings_per_target = get_pings_per_target(database, ping_table, removed_vps)
            # where_clause=" AND dst_addr=toIPv4('213.173.161.11')")

            for geolocation_database_file in [IPINFO_GEOLOCATION_SNAPSHOT, MM_GEOLOCATION_SNAPSHOT]:
                with maxminddb.open_database(geolocation_database_file) as geolocation_reader:
                    for strategy in [
                        STRATEGY_BRUTE_FORCE,
                        STRATEGY_CLOSEST_50,
                        STRATEGY_RANDOM_50_CC,

                    ]:
                        if strategy == STRATEGY_BRUTE_FORCE and geolocation_database_file == MM_GEOLOCATION_SNAPSHOT:
                            continue
                        shortest_ping_per_target = geolocate(targets, pings_per_target, vps, geolocation_reader, geoloc_per_vp,
                                      strategy=strategy, geolocation_database=str(geolocation_database_file), table=ping_table)

                        if ping_table == MESHED_PING_TABLE:
                            # Compute error
                            Y_error = []
                            for target, (min_vp, min_rtt) in shortest_ping_per_target.items():
                                if target in geoloc_per_vp and min_vp in geoloc_per_vp:
                                    error_distance = haversine(geoloc_per_vp[min_vp], geoloc_per_vp[target])
                                    Y_error.append(error_distance)
                                else:
                                    if target not in geoloc_per_vp:
                                        print(f"Target {target} not in geoloc_per_vp")
                                    elif min_vp not in geoloc_per_vp:
                                        print(f"min vp {min_vp} {min_rtt} not in geoloc_per_vp")
                            Ys.append(Y_error)
                        elif ping_table == ROUTER_PING_TABLE:
                            Y_min_rtt = []
                            for target, (min_vp, min_rtt) in shortest_ping_per_target.items():
                                if min_vp is None:
                                    continue
                                Y_min_rtt.append(min_rtt)
                            Ys.append(Y_min_rtt)

                    # Add flat geolocation of geo databases
                    if ping_table == MESHED_PING_TABLE:

                        error_distance_cdf = []

                        geoloc_per_target = geolocate_db(pings_per_target.keys(), geolocation_reader, str(geolocation_database_file))

                        for target, (lat, lon, cc) in geoloc_per_target.items():
                            # Compute distance
                            if target not in geoloc_per_vp:
                                continue
                            error_distance = haversine(geoloc_per_vp[target], (lat, lon))
                            error_distance_cdf.append(error_distance)

                        Ys.append(error_distance_cdf)

            dump_json(Ys, Path(ofile))
        else:
            Ys = load_json(Path(ofile))
            # Ys_copy = list(_ for _ in Ys)
            # Ys_copy[1] = list(Ys[3])
            # Ys_copy[2] = list(Ys[4])
            # Ys_copy[3] = list(Ys[1])
            # Ys_copy[4] = list(Ys[2])
            # dump_json(Ys_copy, Path(ofile))



        if ping_table == MESHED_PING_TABLE:

            labels = [
                "Shortest Ping (All probes)",
                "Closest 50 (IPinfo)",
                "Random 50 in country (IPinfo)",
                "IPinfo",
                "Closest 50 (MaxMind)",
                "Random 50 in country (MaxMind)",
                "MaxMind"
            ]

            for i in range(0, len(labels)):
                metro_level = len([y for y in Ys[i] if y <= 40])
                print(labels[i], metro_level, len(Ys[i]), metro_level / len(Ys[i]))

            fig, ax = plot_multiple_cdf(Ys, 10000, 1, 10000,
                                        "Error distance (km)",
                                        "CDF of targets",
                                        xscale="log",
                                        yscale="linear",
                                        legend=labels)

            homogenize_legend(ax, "lower right", legend_size=11)
            ofile = f"resources/geolocation_databases_error.pdf"

        elif ping_table == ROUTER_PING_TABLE:
            labels = [
                "Shortest Ping (All probes)",
                "Closest 50 (IPinfo)",
                "Random 50 in country (IPinfo)",
                "Closest 50 (MaxMind)",
                "Random 50 in country (MaxMind)",
            ]

            for i in range(0, len(labels)):
                metro_level = len([y for y in Ys[i] if y <= 2])
                print(labels[i], metro_level, len(Ys[i]), metro_level / len(Ys[i]))

            fig, ax = plot_multiple_cdf(Ys, 10000, 1, max(max(Y for Y in Ys)),
                                        "Min RTT (ms)",
                                        "CDF of targets",
                                        xscale="log",
                                        yscale="linear",
                                        legend=labels)

            homogenize_legend(ax, "lower right", legend_size=11)
            ofile = f"resources/geolocation_databases_min_rtt.pdf"

        plot_save(ofile, is_tight_layout=True)

if __name__ == "__main__":
    main()
