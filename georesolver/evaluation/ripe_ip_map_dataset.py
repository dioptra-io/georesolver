"""evaluate ripe IP map coverage"""

from loguru import logger
from ipaddress import IPv4Network, AddressValueError

from georesolver.common.files_utils import load_csv
from georesolver.common.settings import PathSettings

path_settings = PathSettings()


def main() -> None:
    """entry point for ripe"""
    rows = load_csv(path_settings.DATASET / "ripe_ip_map/geolocations_2024-12-10.csv")

    ripe_ip_map_geoloc = []
    for row in rows:
        geoloc_target = row.split(",")[0]
        try:
            network_target = IPv4Network(geoloc_target)
        except AddressValueError:
            continue

        netmask = geoloc_target.split("/")[-1]
        network_ip_addr = geoloc_target.split("/")[0]

        if netmask != "32":
            print(network_target)
            for host in network_target.hosts():
                ripe_ip_map_geoloc.append(host)
        else:
            ripe_ip_map_geoloc.append(network_ip_addr)

    logger.info(f"Nb IPv4 IP addresses in RIPE IP map:: {len(ripe_ip_map_geoloc)}")


if __name__ == "__main__":
    main()
