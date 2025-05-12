"""georesolver is IPv4, now lets do IPv6"""

import asyncio

from pyasn import pyasn
from loguru import logger
from pych_client import ClickHouseClient

from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.clickhouse import CreateIPv6VPsTable, InsertCSV
from georesolver.clickhouse.queries import get_tables, load_vps, get_hostnames
from georesolver.prober.ripe_api import RIPEAtlasAPI
from georesolver.common.ip_addresses_utils import (
    get_host_ip_addr,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.files_utils import create_tmp_csv_file, load_json, dump_csv
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()

VPS_RAW_TABLE_IPV6 = "vps_raw_table_ipv6"


async def retrieve_vps() -> list:
    """return all RIPE Atlas VPs (set probe_only to remove anchors)"""
    vps = []
    rejected = 0
    vp: dict = None
    async for vp in RIPEAtlasAPI().get_raw_vps():
        if (
            vp["status"]["name"] != "Connected"
            or vp.get("geometry") is None
            or vp.get("address_v4") is None
            or vp.get("address_v6") is None
            or vp.get("asn_v4") is None
            or vp.get("country_code") is None
            or RIPEAtlasAPI().is_geoloc_disputed(vp)
        ):
            rejected += 1
            continue

        reduced_vp = {
            "address_v4": vp["address_v4"],
            "address_v6": vp["address_v6"],
            "asn_v4": vp["asn_v4"],
            "asn_v6": vp["asn_v6"],
            "country_code": vp["country_code"],
            "geometry": vp["geometry"],
            "lat": vp["geometry"]["coordinates"][1],
            "lon": vp["geometry"]["coordinates"][0],
            "id": vp["id"],
            "is_anchor": vp["is_anchor"],
        }

        vps.append(reduced_vp)
        break

    logger.info(f"Retrieved {len(vps)} VPs connected on RIPE Atlas")
    logger.info(f"VPs removed: {rejected}")
    logger.info(f"Number of Probes  = {len([vp for vp in vps if not vp['is_anchor']])}")
    logger.info(f"Number of Anchors = {len([vp for vp in vps if vp['is_anchor']])}")

    return vps


def parse_vp(vp, asndb) -> str:
    """parse RIPE Atlas VPs data for clickhouse insertion"""

    _, bgp_prefix = route_view_bgp_prefix(vp["address_v6"], asndb)
    _, bgp_prefix = route_view_bgp_prefix(vp["address_v6"], asndb)
    subnet_v4 = get_prefix_from_ip(vp["address_v4"])
    subnet_v6 = get_prefix_from_ip(vp["address_v6"], ipv6=True)

    return f"""{vp['address_v4']},\
        {vp['address_v6']},\
        {subnet_v4},\
        {subnet_v6},\
        {vp['asn_v4']},\
        {bgp_prefix},\
        {vp['country_code']},\
        {vp['lat']},\
        {vp['lon']},\
        {vp['id']},\
        {vp['is_anchor']}"""


def insert_vps(vps: list[dict], output_table: str) -> None:
    """insert IPv6 vps"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    csv_data = []
    for vp in vps:
        parsed_vp = parse_vp(vp, asndb)
        csv_data.append(parsed_vp)

    tmp_file_path = create_tmp_csv_file(csv_data)

    with ClickHouseClient(**ch_settings.clickhouse) as client:
        CreateIPv6VPsTable().execute(client, output_table)
        InsertCSV().execute(
            client=client,
            table_name=output_table,
            data=tmp_file_path.read_bytes(),
        )

    tmp_file_path.unlink()


async def get_vps(input_table: str) -> list[dict]:
    """get IPv6 VPs"""
    tables = get_tables()
    if not input_table in tables:
        vps = await retrieve_vps()
        insert_vps(vps, input_table)
        return vps
    else:
        vps = load_vps(input_table)


def select_hostnames() -> list[str]:
    """select hostnames that support IPv6"""
    host_addr = get_host_ip_addr(ipv6=True)
    host_subnet = get_prefix_from_ip(host_addr, ipv6=True)
    ecs_hostnames_path = path_settings.HOSTNAMES_GEORESOLVER

    asyncio.run(
        run_dns_mapping(
            subnets=[host_subnet],
            hostname_file=ecs_hostnames_path,
            request_type="AAAA",
            output_table=ch_settings.VPS_ECS_MAPPING_TABLE + "_ipv6",
            ipv6=True,
        )
    )


def get_ipV6_ecs_hostnames(input_table: str) -> list[str]:
    """perform and/or return ECS hostnames in IPv6"""
    ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ipv6_ECS_hostnames.csv"

    tables = get_tables()

    logger.info(f"Iterative ECS hostnames table:: {input_table}")

    if input_table not in tables:
        # load hostnames per org per ns
        best_hostnames_per_org_per_ns = load_json(
            path_settings.HOSTNAME_FILES / "best_hostnames_per_org_per_ns.json"
        )

        ecs_hostnames = set()
        for ns, hostnames_per_org in best_hostnames_per_org_per_ns.items():
            for hostnames in hostnames_per_org.values():
                ecs_hostnames.update([h for _, h, _, _ in hostnames])

        logger.info(f"Found {len(ecs_hostnames)} hostnames supporting ECS")

        dump_csv(ecs_hostnames, ecs_hostnames_path)

        host_addr = get_host_ip_addr()
        host_subnet = get_prefix_from_ip(host_addr)

        asyncio.run(
            run_dns_mapping(
                subnets=[host_subnet],
                hostname_file=ecs_hostnames_path,
                output_table=input_table,
                request_type="AAAA",
                ipv6=True,
            )
        )

    # load hostnames from table
    iterative_ecs_hostnames = get_hostnames(input_table)

    logger.info(
        f"Retrieved {len(iterative_ecs_hostnames)} hostnames supporting ECS with ZDNS resolver"
    )

    return iterative_ecs_hostnames


def run_ipV6_meshed_pings() -> None:
    """run meshed pings between IPv6 vps and targets"""
    pass


def evaluation() -> None:
    """geolocation error on RIPE Atlas targets"""
    pass


def ipV6_sample() -> None:
    """run IPv6 georesolver on a sample of IPv6 addrs"""
    pass


def main() -> None:
    """
    entrypoint:
        - select vps and targets IPv6
        - perform meshed pings in IPv6
        - get hostnames that support and return IPv6 address (with ECS)
        - VP selection using /56 subnets
        - no last point, we should be done by now
    """

    # 1. get vps
    vps = asyncio.run(get_vps(VPS_RAW_TABLE_IPV6))
    select_hostnames()


if __name__ == "__main__":
    main()
