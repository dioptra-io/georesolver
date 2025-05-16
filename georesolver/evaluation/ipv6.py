"""georesolver is IPv4, now lets do IPv6"""

import asyncio

from tqdm import tqdm
from pyasn import pyasn
from loguru import logger
from collections import defaultdict
from datetime import datetime, timedelta
from pych_client import ClickHouseClient

from georesolver.prober import RIPEAtlasProber
from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.clickhouse import CreateIPv6VPsTable, InsertCSV
from georesolver.clickhouse.queries import (
    load_vps,
    get_tables,
    get_targets,
    load_targets,
    get_hostnames,
    get_measurement_ids,
    get_pings_per_src_dst,
)
from georesolver.prober.ripe_api import RIPEAtlasAPI
from georesolver.agent.insert_process import retrieve_pings, retrieve_traceroutes
from georesolver.common.geoloc import (
    haversine,
    filter_default_country_geolocation,
    compute_remove_wrongly_geolocated_probes,
)
from georesolver.common.ip_addresses_utils import (
    get_host_ip_addr,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.files_utils import (
    dump_csv,
    load_json,
    dump_json,
    create_tmp_csv_file,
)
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    ConstantSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
constant_settings = ConstantSettings()

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
    else:
        logger.info(f"{hostname_ecs_table=} already exists, skipping ECS filtering")


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

    else:
        logger.info(f"Table {input_table} already exists, skipping ECS filtering")

    # load hostnames from table
    iterative_ecs_hostnames = get_hostnames(input_table)

    logger.info(
        f"Retrieved {len(iterative_ecs_hostnames)} hostnames supporting ECS with ZDNS resolver"
    )

    return iterative_ecs_hostnames


def meshed_pings_schedule(vps_table: str, update_meshed_pings: bool = True) -> dict:
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


def get_vp_distance_matrix(vp_coordinates: dict, target_ids: list[str]) -> dict:
    """compute pairwise distance between each VPs and a list of target ids (RIPE Atlas anchors)"""
    logger.info(
        f"Calculating VP distance matrix for {len(vp_coordinates)}x{len(target_ids)}"
    )

    vp_distance_matrix = defaultdict(dict)
    # for better efficiency only compute pairwise distance for the targets
    for vp_i_id in tqdm(target_ids):
        vp_i_coordinates = vp_coordinates[vp_i_id]

        for vp_j_id, vp_j_coordinates in vp_coordinates.items():
            if vp_i_id == vp_j_id:
                continue

            distance = haversine(vp_i_coordinates, vp_j_coordinates)

            vp_distance_matrix[vp_i_id][vp_j_id] = distance
            vp_distance_matrix[vp_j_id][vp_i_id] = distance

    return vp_distance_matrix


def filter_wrongful_geolocation(
    targets: list[dict], vps: list[dict], meshed_pings_table: str
) -> None:
    """remove VPs based on SOI violation condition"""
    target_ids = [target["id"] for target in targets]

    vp_per_id = {}
    vp_per_addr = {}
    vp_coordinates = {}
    for vp in vps:
        vp_per_id[str(vp["id"])] = vp
        vp_per_addr[vp["addr"]] = vp
        vp_coordinates[str(vp["id"])] = vp["lat"], vp["lon"]

    if not path_settings.VPS_PAIRWISE_DISTANCE_IPv6.exists():
        vp_distance_matrix = get_vp_distance_matrix(vp_coordinates, target_ids)
        dump_json(vp_distance_matrix, path_settings.VPS_PAIRWISE_DISTANCE_IPv6)

    if (path_settings.DATASET / "wrongly_geolocated_probes_ipv6.json").exists():
        vp_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE_IPv6)

        ping_per_target_per_vp = get_pings_per_src_dst(
            meshed_pings_table, threshold=300
        )

        removed_probes = compute_remove_wrongly_geolocated_probes(
            ping_per_target_per_vp=ping_per_target_per_vp,
            vp_coordinates=vp_coordinates,
            vp_distance_matrix=vp_distance_matrix,
            vp_per_addr=vp_per_addr,
            vp_per_id=vp_per_id,
        )

        dump_json(
            removed_probes,
            path_settings.DATASET / "wrongly_geolocated_probes_ipv6.json",
        )

        logger.info(f"Removing {len(removed_probes)} probes")

    return removed_probes


async def filter_vps(vps_table: str, pings_table: str) -> None:
    """filter VPs based on default geoloc and SOI violatio"""
    targets = load_targets(vps_table)
    vps = load_vps(vps_table)

    ping_filtered = filter_wrongful_geolocation(targets, vps, pings_table)
    country_filtered = filter_default_country_geolocation(
        vps, path_settings.DATASET / "country_filtered_vps_ipv6.json"
    )

    removed_vps = list(set(ping_filtered).union(set(country_filtered)))

    dump_json(removed_vps, path_settings.DATASET / "removed_vps_ipv6.json")

    logger.info(f"{len(removed_vps)} VPs removed")

    return removed_vps


def meshed_traceroutes_schedule(vps_table: str, max_nb_traceroutes: int = 100) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    vps = load_vps(vps_table)
    vps_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE_IPv6)

    # get vp id per addr
    vp_per_id = {}
    vp_per_addr = {}
    for vp in vps:
        vp_per_id[str(vp["id"])] = vp
        vp_per_addr[vp["addr"]] = vp

    # parse distance matrix
    ordered_distance_matrix = {}
    for vp in vps:
        distances = vps_distance_matrix[str(vp["id"])]
        ordered_distance_matrix[str(vp["id"])] = sorted(
            distances.items(), key=lambda x: x[-1]
        )

    traceroute_targets_per_vp = defaultdict(list)
    for vp in vps:
        closest_vp_ids = ordered_distance_matrix[str(vp["id"])]
        closest_vp_addrs = [vp_per_id[id]["addr"] for id, _ in closest_vp_ids]
        # only take targets outside of VPs subnet for mesehd traceroutes
        i = 0
        for addr in closest_vp_addrs:
            if get_prefix_from_ip(addr) != vp["subnet"]:
                traceroute_targets_per_vp[vp["id"]].append(addr)
                i += 1

            if i > max_nb_traceroutes:
                break

    # group VPs that must trace the same target to maximize parallel measurements
    traceroute_schedule = defaultdict(set)
    for vp_id, closest_vp_addrs in traceroute_targets_per_vp.items():
        for addr in closest_vp_addrs:
            traceroute_schedule[addr].add(vp_id)

    logger.info(
        f"Traceroute Schedule for:: {len(traceroute_schedule.values())} targets"
    )

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    measurement_schedule = []
    for target, vps in traceroute_schedule.items():
        vps = list(vps)
        for i in range(0, len(vps), batch_size):
            batch_vps = vps[i : (i + batch_size)]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target,
                    [id for id in batch_vps],
                )
            )

    return measurement_schedule


async def insert_measurements(
    measurement_schedule: list[tuple],
    probing_tags: list[str],
    output_table: str,
    measurement_type: str,
    wait_time: int = 60,
    only_once: bool = False,
) -> None:
    """insert measurement once they are tagged as Finished on RIPE Atlas"""
    current_time = datetime.timestamp(datetime.now() - timedelta(days=7))
    cached_measurement_ids = set()

    logger.info(f"Starting inserting measurements {probing_tags=}; {measurement_type=}")

    while True:
        # load measurement finished from RIPE Atlas
        stopped_measurement_ids = await RIPEAtlasAPI().get_stopped_measurement_ids(
            start_time=current_time, tags=probing_tags
        )

        # load already inserted measurement ids
        inserted_ids = get_measurement_ids(output_table)

        # stop measurement once all measurement are inserted
        all_measurement_ids = set(inserted_ids).union(cached_measurement_ids)
        if len(all_measurement_ids) >= len(measurement_schedule):
            logger.info(
                f"All measurement inserted:: {len(inserted_ids)=}; {len(measurement_schedule)=}"
            )
            break

        measurement_to_insert = set(stopped_measurement_ids).difference(
            set(inserted_ids)
        )

        # check cached measurements,
        # some measurement are not insersed because no results
        measurement_to_insert = set(measurement_to_insert).difference(
            cached_measurement_ids
        )

        logger.info(f"{len(stopped_measurement_ids)=}")
        logger.info(f"{len(inserted_ids)=}")
        logger.info(f"{len(measurement_to_insert)=}")

        if not measurement_to_insert:
            await asyncio.sleep(wait_time)
            continue

        # insert measurement
        batch_size = 1_000
        for i in range(0, len(measurement_to_insert), batch_size):

            logger.info(
                f"Batch {i // batch_size}/{len(measurement_to_insert) // batch_size}"
            )
            batch_measurement_ids = list(measurement_to_insert)[i : i + batch_size]

            if measurement_type == "ping":

                await retrieve_pings(
                    batch_measurement_ids, output_table, step_size=3, ipv6=True
                )
            elif measurement_type == "traceroute":
                await retrieve_traceroutes(
                    batch_measurement_ids, output_table, step_size=3, ipv6=True
                )
            else:
                raise RuntimeError(f"Unknwo measurement type:: {measurement_type}")

        cached_measurement_ids.update(measurement_to_insert)
        current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

        if only_once:
            break

        await asyncio.sleep(wait_time)


async def run_measurement(
    measurement_schedule: list[tuple[str, int]],
    measurement_tag: str,
    output_table: str,
    measurement_type: str,
    check_cache: bool = False,
) -> None:
    """
    perform pings towards one IP address
    for each /24 prefix present in VPs ECS mapping redirection
    """
    # in case measurement failed previously
    if check_cache:
        # retrieve previously ongoing measurements
        await insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", measurement_tag],
            output_table=output_table,
            measurement_type=measurement_type,
            only_once=True,
        )

        logger.info("Filtering measurement schedule")
        logger.info(f"Original schedule: {len(measurement_schedule)} targets")

        # filter measurement schedule
        targets = get_targets(output_table)
        filtered_measurement_schedule = []
        for target_addr, vp_ids in tqdm(measurement_schedule):
            if target_addr in targets:
                continue

            filtered_measurement_schedule.append((target_addr, vp_ids))

        logger.info(f"Filtered schedule: {len(measurement_schedule)} targets")

        measurement_schedule = filtered_measurement_schedule

    prober = RIPEAtlasProber(
        probing_type=measurement_type,
        probing_tag=measurement_tag,
        output_table=output_table,
        protocol="ICMP",
        ipv6=True,
    )

    await asyncio.gather(
        prober.main(measurement_schedule),
        insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", measurement_tag],
            output_table=output_table,
            measurement_type=measurement_type,
        ),
    )

    logger.info("Meshed CDNs pings measurement done")


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
    do_meshed_traceroutes: bool = False
    do_evaluation: bool = False

    # 1. get vps
    vps = asyncio.run(get_vps(VPS_RAW_TABLE_IPV6))

    if do_select_hostnames:
        filter_ecs_hostnames(hostname_ecs_table="ecs_hostnames_ipv6")
        select_ecs_hostnames(
            vps_mapping_table=ch_settings.VPS_ECS_MAPPING_TABLE + "_ipv6",
        )
    if do_meshed_pings:
        measurement_schedule = meshed_pings_schedule(VPS_RAW_TABLE_IPV6)
        asyncio.run(
            run_measurement(
                measurement_schedule=measurement_schedule,
                measurement_tag="meshed-pings-ipv6",
                output_table=ch_settings.VPS_MESHED_PINGS_TABLE + "_ipv6",
                measurement_type="ping",
            )
        )

    if do_meshed_traceroutes:
        # filter vps before traceroutes
        asyncio.run(filter_vps())

        measurement_schedule = meshed_traceroutes_schedule(VPS_RAW_TABLE_IPV6)
        asyncio.run(
            run_measurement(
                measurement_schedule=measurement_schedule,
                measurement_tag="meshed-traceroutes-ipv6",
                output_table=ch_settings.VPS_MESHED_TRACEROUTE_TABLE + "_ipv6",
                measurement_type="traceroute",
            )
        )


if __name__ == "__main__":
    main()
