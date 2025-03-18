"""generate geolocation map based on ping table results"""

import typer
import folium

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from folium.plugins import MarkerCluster

from georesolver.clickhouse.queries import (
    get_pings_per_target_extended,
    load_vps,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


def get_shortest_ping(input_table: str, latency_threshold: int) -> tuple[dict]:
    """retrieve georesolver shortest ping"""
    pings_per_target = get_pings_per_target_extended(input_table)

    shortest_ping_per_target = {}
    under_2_ms = {}
    for target, pings in pings_per_target.items():
        vp_addr, vp_id, min_rtt = min(pings, key=lambda x: x[-1])

        shortest_ping_per_target[target] = (vp_addr, vp_id, min_rtt)

        if min_rtt <= latency_threshold:
            under_2_ms[target] = (vp_addr, vp_id, min_rtt)

    return shortest_ping_per_target, under_2_ms


def main(input_table: str, output_file: Path, latency_threhsold: int = 2) -> None:
    """entry point, create a map based on ping or geoloc table"""

    _, under_2_ms_geoloc = get_shortest_ping(input_table, latency_threhsold)
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    vps_per_id = {}
    for vp in vps:
        vps_per_id[vp["id"]] = vp

    m = folium.Map(
        zoom_start=12,
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    )
    marker_cluster = MarkerCluster().add_to(m)

    logger.info(
        f"Retreived {len(under_2_ms_geoloc)} targets under {latency_threhsold} ms"
    )

    for target_addr, (vp_addr, vp_id, min_rtt) in tqdm(under_2_ms_geoloc.items()):
        vp = vps_per_id[vp_id]

        folium.Marker(
            location=[vp["lat"], vp["lon"]],
            popup=f"{target_addr}; {min_rtt}; {vp_id}; {vp_addr}",
            icon=folium.Icon(color="blue", icon=None),
        ).add_to(marker_cluster)

    m.save(output_file)


if __name__ == "__main__":
    typer.run(main)
