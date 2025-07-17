"""Generate Follium geoloc map using ITDK ping results (cosmetic only)"""

import folium
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from folium.plugins import MarkerCluster

from georesolver.clickhouse.queries import (
    get_pings_per_target_extended,
    load_vps,
    get_vps_ids_per_target,
)
from georesolver.common.files_utils import load_csv, dump_csv
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

GEORESOLVER_ITDK_PING_TABLE = "itdk_ping_true_ids"
GEORESOLVER_ITDK_GEOLOC_FILE = path_settings.DATASET / "itdk/itdk_geoloc.csv"
GEORESOLVER_IP_PER_COUNTRY = path_settings.DATASET / "itdk/ips_per_country.csv"


def get_country_name_per_code() -> dict:
    """get all country name based on their country codes"""
    country_data = load_csv(path_settings.COUNTRIES_INFO)
    country_name_per_code = {}
    for row in country_data[1:]:
        row = row.split(",")
        country_name = row[0]
        country_code = row[2]

        country_name_per_code[country_code] = country_name

    return country_name_per_code


def get_georesolver_shortest_ping() -> tuple[dict]:
    """retrieve georesolver shortest ping"""
    initial_targets = load_csv(
        path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv"
    )
    pings_per_target = get_pings_per_target_extended(GEORESOLVER_ITDK_PING_TABLE)

    shortest_ping_per_target = {}
    under_2_ms = {}
    for target in tqdm(initial_targets):
        try:
            pings = pings_per_target[target]
        except KeyError:
            continue

        vp_addr, vp_id, min_rtt = min(pings, key=lambda x: x[-1])

        shortest_ping_per_target[target] = (vp_addr, vp_id, min_rtt)

        if min_rtt <= 2:
            under_2_ms[target] = (vp_addr, vp_id, min_rtt)

    return shortest_ping_per_target, under_2_ms


def create_georesolver_geoloc_file(shortest_pings: dict) -> None:
    country_name_per_code = get_country_name_per_code()
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    vps_per_id = get_vps_ids_per_target(vps)

    rows = ["addr,lat,lon,country_code,country_name"]
    for target_addr, (_, vp_id, _) in tqdm(shortest_pings.items()):
        vp = vps_per_id[vp_id]
        country_name = country_name_per_code[vp["country_code"]]
        rows.append(
            f"{target_addr},{vp['lat']},{vp['lon']},{vp['country_code']},{country_name}"
        )

    dump_csv(rows, GEORESOLVER_ITDK_GEOLOC_FILE)


def get_georesolver_ip_per_country() -> None:
    georesolver_geoloc = load_csv(GEORESOLVER_ITDK_GEOLOC_FILE)

    ip_per_country = defaultdict(int)
    for row in georesolver_geoloc[1:]:
        row = row.split(",")
        country_name = row[-1]
        ip_per_country[country_name] += 1

    all_ips = len(georesolver_geoloc[1:])
    fraction_per_country = {}
    for country, nb_ips in ip_per_country.items():
        fraction_per_country[country] = nb_ips / all_ips

    rows = ["country_name,fraction_of_ips"]
    for country, frac in fraction_per_country.items():
        rows.append(f"{country},{frac}")

    dump_csv(rows, GEORESOLVER_IP_PER_COUNTRY)


def heat_map() -> None:
    georesolver_geoloc = pd.read_csv(GEORESOLVER_IP_PER_COUNTRY)
    political_countries_url = (
        "http://geojson.xyz/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson"
    )

    folium.Choropleth(
        geo_data=political_countries_url,
        data=georesolver_geoloc,
        columns=["country_name", "fraction_of_ips"],
        key_on="feature.properties.name",
    ).add_to(m)


def main() -> None:
    if not GEORESOLVER_ITDK_GEOLOC_FILE.exists():
        _, under_2_ms = get_georesolver_shortest_ping()
        create_georesolver_geoloc_file(under_2_ms)
    if not GEORESOLVER_IP_PER_COUNTRY.exists():
        get_georesolver_ip_per_country()

    m = folium.Map(
        zoom_start=12,
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    )
    marker_cluster = MarkerCluster().add_to(m)

    rows = load_csv(GEORESOLVER_ITDK_GEOLOC_FILE)[1:]
    for row in rows:
        row = row.split(",")
        target_addr = row[0]
        lat, lon = float(row[1]), float(row[2])

        folium.Marker(
            location=[lat, lon],
            popup=f"{target_addr}",
            icon=folium.Icon(color="blue", icon=None),
        ).add_to(marker_cluster)

    m.save("footprint.html")


if __name__ == "__main__":
    main()
