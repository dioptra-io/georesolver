from loguru import logger
from random import sample
from collections import defaultdict
from pathlib import Path
from ipaddress import IPv4Address, IPv4Network

from georesolver.common.files_utils import (
    load_iter_csv,
    dump_csv,
    dump_json,
    load_json,
    load_csv,
)
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

ITDK_PATH = path_settings.DATASET / "itdk/"
ITDK_INTERFACES = ITDK_PATH / "midar-iff.ifaces"
ITDK_ROUTER_PER_INTERFACES = ITDK_PATH / "itdk_router_per_interfaces.json"
ITDK_ADDRS_ALL_PATH = ITDK_PATH / "itdk_addrs_all.csv"
ITDK_RESPONSIVE_ADDRS_ALL_PATH = ITDK_PATH / "itdk_responsive_addrs_all.csv"
ZMAP_ADDRS_PATH = ITDK_PATH / "zmap_icmp_scan_15_11_2024.csv"
ITDK_ROUTER_INTERFACES_PATH = ITDK_PATH / "itdk_router_interfaces.json"
HOIHO_GEOLOC_PATH = ITDK_PATH / "midar-iff.nodes.geo"
HOIHO_GEOLOC_ALL_ADDRS = ITDK_PATH / "hoiho_geoloc_all_addrs.csv"
HOIHO_GEOLOC_PARSED_PATH = ITDK_PATH / "hoiho_parsed_geoloc.json"
ITDK_RESPONSIVE_ROUTER_INTERFACE_PATH = (
    ITDK_PATH / "itdk_responsive_router_interface_parsed.csv"
)


def get_itdk_router_interfaces() -> None:
    """
    extract all IP addresses that belongs to router interfaces:
        - example row : '227.40.95.226 N37253355 L56036288 T'
        - step 1      : only take IP addresses that belong to routers (T)
        - step 2      : remove IANA multicast IP addrs (* in traceroutes)
        - step 3      : regroup IP addresses per node ID (alias resolution)
        - step 4      : dump router interfaces
    """

    itdk_ifaces = load_iter_csv(ITDK_INTERFACES)

    row: str = ""
    router_interfaces = defaultdict(list)
    for row in itdk_ifaces:

        if row.startswith("#"):
            continue

        values = row.split(" ")

        if "T" not in values:
            continue

        node_id = values[1]
        interface_addr = values[0]

        # Remove IANA reserved multicast prefix 224.0.0.0/3
        # These are not real IP addresses,
        # used to infer unresponsive IP addresses associated with a node
        if IPv4Address(interface_addr) in IPv4Network("224.0.0.0/3"):
            continue

        router_interfaces[node_id].append(interface_addr)

    logger.info(f"Found:: {len(router_interfaces)} routers in dataset")

    dump_json(router_interfaces, ITDK_ROUTER_INTERFACES_PATH)


def get_itdk_all_addrs() -> None:
    """simply extract all ITDK interfaces, without router correspondance"""
    itdk_all_addrs = set()
    itdk_router_interfaces = load_json(ITDK_ROUTER_INTERFACES_PATH)

    for _, interfaces in itdk_router_interfaces.items():
        itdk_all_addrs.update(interfaces)

    dump_csv(itdk_all_addrs, ITDK_ADDRS_ALL_PATH)


def get_itdk_router_per_interface() -> None:
    """generate dict, key are router interface and value is the id of the router"""
    itdk_router_per_interfaces = {}
    itdk_router_interfaces = load_json(ITDK_ROUTER_INTERFACES_PATH)

    for router_id, interfaces in itdk_router_interfaces.items():
        for interface in interfaces:
            itdk_router_per_interfaces[interface] = router_id

    dump_json(itdk_router_per_interfaces, ITDK_ROUTER_PER_INTERFACES)


def get_itdk_responsive_all_addrs() -> None:
    """return all responsive IP addrs from the ITDK dataset (without router correspondance)"""
    zmap_addrs = load_csv(ZMAP_ADDRS_PATH)
    itdk_all_addrs = load_csv(ITDK_ADDRS_ALL_PATH)

    logger.info(f"ITDK all addrs:: {len(itdk_all_addrs)}")
    logger.info(f"ZMap addrs:: {len(zmap_addrs)}")

    itdk_responsive_all_addrs = set(itdk_all_addrs).intersection(set(zmap_addrs))

    logger.info(f"ITDK responsive addrs:: {len(itdk_responsive_all_addrs)}")

    dump_csv(itdk_responsive_all_addrs, ITDK_RESPONSIVE_ADDRS_ALL_PATH)


def get_itdk_responsive_router_interface() -> None:
    """for each router of ITDK dataset find one responsive IP addr (if any) using ZMap scan"""
    if not ITDK_ROUTER_INTERFACES_PATH.exists():
        logger.info(f"Generating:: {ITDK_ROUTER_INTERFACES_PATH} file")
        get_itdk_router_interfaces()
    else:
        logger.info(f"File:: {ITDK_ROUTER_INTERFACES_PATH} already exists")

    if not ITDK_ADDRS_ALL_PATH.exists():
        logger.info(f"Generating:: {ITDK_ADDRS_ALL_PATH} file")
        get_itdk_all_addrs()
    else:
        logger.info(f"File:: {ITDK_ADDRS_ALL_PATH} already exists")

    if not ITDK_RESPONSIVE_ADDRS_ALL_PATH.exists():
        logger.info(f"Generating:: {ITDK_RESPONSIVE_ADDRS_ALL_PATH} file")
        get_itdk_responsive_all_addrs()
    else:
        logger.info(f"File:: {ITDK_RESPONSIVE_ADDRS_ALL_PATH} already exists")

    if not ITDK_ROUTER_PER_INTERFACES.exists():
        logger.info(f"Generating:: {ITDK_ROUTER_PER_INTERFACES} file")
        get_itdk_router_per_interface()
    else:
        logger.info(f"File:: {ITDK_ROUTER_PER_INTERFACES} already exists")

    itdk_responsive_all_addrs = load_iter_csv(ITDK_RESPONSIVE_ADDRS_ALL_PATH)
    itdk_router_per_interface = load_json(ITDK_ROUTER_PER_INTERFACES)
    responsive_router_interface = []

    router_ids = set()
    for responsive_addr in itdk_responsive_all_addrs:
        router_id = itdk_router_per_interface[responsive_addr]
        # only take one responsive IP addr per router
        if router_id in router_ids:
            continue

        responsive_router_interface.append(responsive_addr)
        router_ids.add(router_id)

    dump_csv(responsive_router_interface, ITDK_RESPONSIVE_ROUTER_INTERFACE_PATH)


def get_random_itdk_routers(nb_addr: int, out_file: Path, mode: str = "a") -> None:
    """get random IP addresses on which to perform geolocation for testing and demo"""
    dump_csv(
        data=sample(
            [
                row.split(",")[-1]
                for row in load_iter_csv(ITDK_RESPONSIVE_ROUTER_INTERFACE_PATH)
            ],
            nb_addr,
        ),
        output_file=out_file,
        mode=mode,
    )


def parse_hoiho_geoloc() -> None:
    """dump json file with each target geoloc coordinates"""
    hoiho_raw_geoloc = load_csv(HOIHO_GEOLOC_PATH)
    router_ip_address_per_node = load_json(ITDK_ROUTER_INTERFACES_PATH)

    nodes = set()
    targets = set()
    hoiho_parsed_geoloc = {}
    for row in hoiho_raw_geoloc:
        row = row.split("\t")
        geoloc_source = row[-1]

        if geoloc_source != "hoiho":
            continue

        node = row[0].split(" ")[-1][:-1]
        country, lat, lon = row[3], row[5], row[6]

        try:
            target_addrs = router_ip_address_per_node[node]
        except KeyError:
            logger.error(f"Cannot find node:: {node}")
            continue

        targets.update(target_addrs)
        nodes.add(node)

        for target_addr in target_addrs:
            hoiho_parsed_geoloc[target_addr] = {
                "country": country,
                "lat": lat,
                "lon": lon,
            }

    logger.info(f"Nb total addr        :: {len(hoiho_raw_geoloc)}")
    logger.info(f"Nb total nodes Hoiho :: {len(nodes)}")
    logger.info(f"Nb total addrs Hoiho :: {len(targets)}")

    dump_json(hoiho_parsed_geoloc, HOIHO_GEOLOC_PARSED_PATH)


if __name__ == "__main__":
    do_get_dataset: bool = True
    do_parse_hoiho_geoloc: bool = True
    do_get_random: bool = False

    if do_get_dataset:
        get_itdk_responsive_router_interface()

    if do_parse_hoiho_geoloc:
        parse_hoiho_geoloc()

    if do_get_random:
        get_random_itdk_routers(1_000, path_settings.DATASET / "demo_targets.csv")
