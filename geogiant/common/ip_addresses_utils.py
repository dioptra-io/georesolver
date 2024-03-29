import socket
import httpx

from pyasn import pyasn
from pathlib import Path
from typing import Sequence
from ipwhois import IPWhois
from loguru import logger
from ipaddress import (
    IPv4Address,
    IPv4Network,
    ip_network,
)


def get_addr_granularity(client_granularity: str, target: dict, asndb: pyasn) -> str:
    """return the desired target/vp granularity (subnet or bgp prefix)"""
    match client_granularity:
        case "client_subnet":
            target_granularity = get_prefix_from_ip(target["address_v4"])
        case "subnet":
            target_granularity = get_prefix_from_ip(target["address_v4"])
        case "client_bgp_prefix":
            _, target_granularity = route_view_bgp_prefix(
                target["address_v4"],
                asndb,
            )
    return target_granularity


def is_valid_ipv4(ip_addr: str) -> str:
    try:
        ip_addr: IPv4Address = IPv4Address(ip_addr)
        if not ip_addr.is_private:
            return True
    except ValueError:
        pass

    return False


def get_cidr_prefix(addr: str) -> str:
    """return cidr prefix of a given IPv4 addr"""
    w = IPWhois(addr)
    r = w.lookup_rdap(depth=1)

    return r["network"]["cidr"]


def get_prefix_from_ip(addr: str):
    """from an ip addr return /24 prefix"""
    prefix = addr.split(".")[:-1]
    prefix.append("0")
    prefix = ".".join(prefix)
    return prefix


def to_ipv6(ipv4_addr: str) -> str:
    """add ipv6 format to ipv4 for unified storage"""
    return "::ffff:" + ipv4_addr


def subnets(network: IPv4Network, new_prefix: int) -> Sequence[int]:
    """
    Faster version of :py:meth:`ipaddress.IPv4Network.subnets`.
    Returns only the network address as an integer.

    >>> from ipaddress import ip_network
    >>> list(subnets(ip_network("0.0.0.0/0"), new_prefix=2))
    [0, 1073741824, 2147483648, 3221225472]
    >>> subnets(ip_network("0.0.0.0/32"), new_prefix=24)
    Traceback (most recent call last):
        ...
    ValueError: new prefix must be longer
    """
    if new_prefix < network.prefixlen:
        raise ValueError("new prefix must be longer")
    start = int(network.network_address)
    end = int(network.broadcast_address) + 1
    step = (int(network.hostmask) + 1) >> (new_prefix - network.prefixlen)
    return range(start, end, step)


def generate_subnets(start_subnet: str, new_prefix: int, out_file_dir: Path) -> None:
    """generate all /24 subnets"""
    subnet_list = list(subnets(ip_network(start_subnet), new_prefix=new_prefix))

    logger.info(f"List generated: {len(subnet_list)} subnets generated")

    subnet: IPv4Network = None
    with out_file_dir.open("w") as f:
        for subnet in subnet_list:
            subnet = IPv4Network(subnet)
            subnet._prefixlen = new_prefix

            if not subnet.is_reserved:
                f.write(str(subnet) + "\n")


def get_host_ip_addr() -> str:
    """get current public IP addr"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_addr = str(s.getsockname()[0])

    # check that we got a public IPv4 addr

    if IPv4Address(ip_addr) and not IPv4Address(ip_addr).is_private:
        return ip_addr
    else:
        raise RuntimeError("could not retrieve user IPv4 addr")


async def ripe_stat_bgp_prefix(ip_addr: str) -> str:
    """return the announced BGP prefix of a given IP addr"""
    base_url = "https://stat.ripe.net/data/prefix-overview/data.json"
    params = {"resource": ip_addr}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url=base_url, params=params).json()
        data = resp["data"]
        bgp_prefix = data["resource"]

    return bgp_prefix


def ripe_stat_bgp_prefix(ip_addr: str) -> tuple[int, str]:
    """in case an IP addr is not in route view dataset, rely on RIPEStat db"""
    asn, bgp_prefix = None, None

    base_url = "https://stat.ripe.net/data/prefix-overview/data.json"
    params = {"resource": ip_addr}

    resp = httpx.get(url=base_url, params=params).json()
    data = resp["data"]

    try:
        bgp_prefix = data["resource"]
        asn = data["asns"][0]["asn"]

    except KeyError:
        logger.error(f"{ip_addr}::Could not resolve BGP prefix")
        pass

    return asn, bgp_prefix


def route_view_bgp_prefix(ip_addr: str, asndb) -> tuple[int, str]:
    """py-asn lookup on route view RIB table dump"""
    asn, bgp_prefix = None, None
    try:
        asn, bgp_prefix = asndb.lookup(ip_addr)
    except IndexError:
        asn, bgp_prefix = ripe_stat_bgp_prefix(ip_addr)

    return asn, bgp_prefix


def get_host_ip_addr() -> str:
    """get current public IP addr"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_addr = str(s.getsockname()[0])

    # check that we got a public IPv4 addr

    if IPv4Address(ip_addr) and not IPv4Address(ip_addr).is_private:
        return ip_addr
    else:
        raise RuntimeError("could not retrieve user IPv4 addr")


# generate_subnets("0.0.0.0/0", 24, DATASET_DIR / "all_24_subnets.csv")
