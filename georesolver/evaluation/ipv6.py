"""georesolver is IPv4, now lets do IPv6"""

import asyncio

from pyasn import pyasn
from loguru import logger
from pych_client import ClickHouseClient

from georesolver.prober import RIPEAtlasProber
from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.clickhouse import CreateIPv6VPsTable, InsertCSV
from georesolver.clickhouse.queries import (
    get_tables,
    load_vps,
    load_targets,
    get_hostnames,
)
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


def filter_ecs_hostnames(hostname_ecs_table: str) -> list[str]:
    """select hostnames that support IPv6"""
    tables = get_tables()
    if not hostname_ecs_table in tables:
        host_addr = get_host_ip_addr(ipv6=True)
        host_subnet = get_prefix_from_ip(host_addr, ipv6=True)
        ecs_hostnames_path = path_settings.HOSTNAME_FILES / "hostnames_1M.csv"

        asyncio.run(
            run_dns_mapping(
                subnets=[host_subnet],
                hostname_file=ecs_hostnames_path,
                request_type="AAAA",
                output_table=hostname_ecs_table,
                ipv6=True,
            )
        )


def select_ecs_hostnames(
    vps_ecs_mapping_table: str,
    hostname_ecs_table: str,
    vps_table: str,
) -> list[str]:
    """select a subset of ECS-DNS hostnames for calculating DNS resolution similarities"""
    tables = get_tables()
    if vps_ecs_mapping_table not in tables:
        vps = load_vps(vps_table)
        vps_subnet = list(set([vp["subnet"] for vp in vps]))
        ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ecs_hostnames_ipv6.csv"

        if not ecs_hostnames_path.exists():
            # get ECS hostnames and dump csv
            hostnames = get_hostnames(hostname_ecs_table)
            dump_csv(ecs_hostnames_path, hostnames)

        asyncio.run(
            run_dns_mapping(
                subnets=vps_subnet,
                hostname_file=ecs_hostnames_path,
                request_type="AAAA",
                output_table=vps_ecs_mapping_table,
                ipv6=True,
            )
        )

    # if table exists, select geographically sparsed VPs
    # get each hostnames hosting/ns organisation
    # calculate redirection score
    # select N hostnames per pair (NS/org), sorted by redirection score
    # return


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


async def meshed_pings_schedule(
    vps_table: str, update_meshed_pings: bool = True
) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    measurement_schedule = []
    cached_meshed_ping_vps = None
    vps = load_vps(vps_table, ipv6=True)
    targets = load_targets(vps_table, ipv6=True)

    logger.info(
        f"Ping Schedule for:: {len(targets)} targets, {len(vps)} VPs per target"
    )

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    for target in targets:
        # filter vps based on their ID and if they already pinged the target
        if cached_meshed_ping_vps and update_meshed_pings:
            try:
                cached_vp_ids = cached_meshed_ping_vps[target["addr"]]
            except KeyError:
                cached_vp_ids = []

            vp_ids = [vp["id"] for vp in vps]
            filtered_vp_ids = list(set(vp_ids).symmetric_difference(set(cached_vp_ids)))
        else:
            filtered_vp_ids = [vp["id"] for vp in vps]

        for i in range(0, len(filtered_vp_ids), batch_size):
            batch_vps = filtered_vp_ids[i : (i + batch_size)]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target["addr"],
                    [vp_id for vp_id in batch_vps if vp_id != target["id"]],
                )
            )

    logger.info(f"Total number of measurement schedule:: {len(measurement_schedule)}")

    return measurement_schedule


async def meshed_pings() -> None:
    """run meshed pings between IPv6 vps and targets"""
    pings_schedule = await meshed_pings_schedule(False)
    output_path = await RIPEAtlasProber(
        probing_type="ping",
        probing_tag="meshed-pings",
        output_table=ch_settings.VPS_MESHED_PINGS_TABLE,
    ).main(pings_schedule)


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
    do_select_hostnames: bool = False
    do_meshed_pings: bool = True
    do_evaluation: bool = False

    # 1. get vps
    vps = asyncio.run(get_vps(VPS_RAW_TABLE_IPV6))

    if do_select_hostnames:
        filter_ecs_hostnames(hostname_ecs_table="ecs_hostnames_ipv6")
        select_ecs_hostnames(
            vps_mapping_table=ch_settings.VPS_ECS_MAPPING_TABLE + "_ipv6",
        )
    if do_meshed_pings:
        meshed_pings(ch_settings.VPS_MESHED_PINGS_TABLE + "_ipv6")
        meshed_traceroutes(ch_settings.VPS_MESHED_TRACEROUTE_TABLE + "_ipv6")


if __name__ == "__main__":
    main()
