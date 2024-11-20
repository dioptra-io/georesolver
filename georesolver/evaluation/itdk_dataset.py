from loguru import logger
from random import sample
from collections import defaultdict
from pathlib import Path
from ipaddress import IPv4Address, IPv4Network

from georesolver.common.files_utils import (
    load_iter_csv,
    dump_csv,
    dump_json,
    load_pickle,
    dump_pickle,
    load_json,
    load_csv,
)
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

HOIHO_SOURCE_NAME = "hoiho"
MAXMIND_SOURCE_NAME = "maxmind"

ITDK_ROUTER_INTERFACES_PATH = path_settings.DATASET / "itdk/itdk_router_interfaces.json"
ITDK_RESPONSIVE_ROUTER_INTERFACE_PATH = (
    path_settings.DATASET / "itdk/itdk_responsive_router_interface_parsed.csv"
)
ITDK_ROUTER_PER_INTERFACES = (
    path_settings.DATASET / "itdk/itdk_router_per_interfaces.json"
)
ITDK_ADDRS_ALL_PATH = path_settings.DATASET / "itdk/itdk_addrs_all.csv"
ITDK_RESPONSIVE_ADDRS_ALL_PATH = (
    path_settings.DATASET / "itdk/itdk_responsive_addrs_all.csv"
)
ZMAP_ADDRS_PATH = path_settings.DATASET / "itdk/zmap_icmp_scan_15_11_2024.csv"


def itdk_geo_dataset_eval() -> None:
    """get proportion of nodes/addrs/subnet for Hoio and overall, compare with Zmap dataset"""
    row: str = ""

    # load every addr per nodes
    if not (path_settings.DATASET / "itdk_addr_nodes.pickle").exists():
        itdk_nodes = load_iter_csv(
            path_settings.DATASET / "static_files/midar-iff.nodes"
        )
        itdk_addr_nodes = defaultdict(set)
        for row in itdk_nodes:
            if row.startswith("node"):
                # get a random IP address for a node
                node = row.split(":")[0].split(" ")[-1]  # Ni
                addrs = row.split(":")[-1].split(" ")
                addrs = [addr.strip("\n") for addr in addrs if addr]

                itdk_addr_nodes[node].update(addrs)

        dump_pickle(itdk_addr_nodes, path_settings.DATASET / "itdk_addr_nodes.pickle")

    # get each nodes geo source
    if not (path_settings.DATASET / "itdk_addr_nodes.pickle").exists():

        itdk_geo_nodes = load_iter_csv(
            path_settings.DATASET / "static_files/midar-iff.nodes.geo"
        )

        itdk_addr_geo_nodes = defaultdict(set)
        for row in itdk_geo_nodes:
            if row.startswith("node.geo"):
                node = row.split(":")[0].split(" ")[-1]  # Ni
                source = row.split(":")[-1].split("\t")[-1]  # Hoio or Maxmind
                itdk_addr_geo_nodes[source].add(node)

        dump_pickle(
            itdk_addr_geo_nodes, path_settings.DATASET / "itdk_addr_geo_nodes.pickle"
        )

    itdk_addr_nodes = load_pickle(path_settings.DATASET / "itdk_addr_nodes.pickle")
    itdk_addr_geo_nodes = load_pickle(
        path_settings.DATASET / "itdk_addr_geo_nodes.pickle"
    )

    hoio_nb_nodes = len(itdk_addr_geo_nodes[HOIHO_SOURCE_NAME])
    maxmind_nb_nodes = len(itdk_addr_geo_nodes[MAXMIND_SOURCE_NAME])

    total_nb_nodes = hoio_nb_nodes + maxmind_nb_nodes
    proportion_hoio_nodes = round(hoio_nb_nodes * 100 / total_nb_nodes, 2)

    logger.info(f"Total nb nodes:: {total_nb_nodes}")
    logger.info(f"Hoio nb nodes:: {hoio_nb_nodes}")
    logger.info(f"Maxmind nb nodes:: {maxmind_nb_nodes}")
    logger.info(
        f"Proportion of Hoio nodes from geo dataset:: {proportion_hoio_nodes} [%]"
    )

    if not (path_settings.DATASET / "hoio_addrs.csv").exists():
        hoio_addrs = set()
        for node in itdk_addr_geo_nodes[HOIHO_SOURCE_NAME]:
            addrs = itdk_addr_nodes[node]
            hoio_addrs.update(addrs)

        dump_csv(hoio_addrs, path_settings.DATASET / "hoio_addrs.csv")

    if not (path_settings.DATASET / "itdk_addrs.csv").exists():
        itdk_addrs = set()
        for node in itdk_addr_geo_nodes[HOIHO_SOURCE_NAME]:
            addrs = itdk_addr_nodes[node]
            itdk_addrs.update(addrs)

        for node in itdk_addr_geo_nodes[MAXMIND_SOURCE_NAME]:
            addrs = itdk_addr_nodes[node]
            itdk_addrs.update(addrs)

        dump_csv(itdk_addrs, path_settings.DATASET / "itdk_addrs.csv")

    hoio_addrs = load_csv(path_settings.DATASET / "hoio_addrs.csv")
    itdk_addrs = load_csv(path_settings.DATASET / "itdk_addrs.csv")

    logger.info(f"Total itdk addrs:: {len(itdk_addrs)}")
    logger.info(f"Hoio addrs:: {len(hoio_addrs)}")
    logger.info(f"Maxmind addrs:: {len(itdk_addrs) - len(hoio_addrs)}")
    logger.info(
        f"Proportion Hoio addrs:: {round(len(hoio_addrs) * 100 / len(hoio_addrs), 2)} [%]"
    )

    # TODO: compare with ZMap
    if not (path_settings.DATASET / "responsive_hoio_addrs.csv").exists():

        zmap_addrs = load_csv(path_settings.DATASET / "zmap_icmp_scan_2024_09.csv")

        responsive_hoio_addrs = set(hoio_addrs).intersection(set(zmap_addrs))
        unresponsive_hoio_addrs = set(hoio_addrs).difference(set(zmap_addrs))

        dump_csv(
            responsive_hoio_addrs,
            path_settings.DATASET / "responsive_hoio_addrs.csv",
        )

        dump_csv(
            unresponsive_hoio_addrs,
            path_settings.DATASET / "unresponsive_hoio_addrs.csv",
        )

    responsive_itdk_addrs = set(itdk_addrs).intersection(set(zmap_addrs))
    unresponsive_itdk_addrs = set(itdk_addrs).difference(set(zmap_addrs))


def get_itdk_router_interfaces() -> None:
    """extract all IP addresses that belongs to router interfaces"""

    itdk_ifaces = load_iter_csv(path_settings.DATASET / "static_files/midar-iff.ifaces")

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


if __name__ == "__main__":
    do_get_dataset: bool = True
    do_get_random: bool = False

    if do_get_dataset:
        get_itdk_responsive_router_interface()

    if do_get_random:
        get_random_itdk_routers(1_000, path_settings.DATASET / "demo_targets.csv")
