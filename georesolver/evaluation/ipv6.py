"""georesolver is IPv4, now lets do IPv6"""

import json
import asyncio
import numpy as np

from tqdm import tqdm
from pyasn import pyasn
from pathlib import Path
from random import sample
from loguru import logger
from ipaddress import IPv6Address
from datetime import datetime, timedelta
from pych_client import ClickHouseClient
from multiprocessing import Pool, cpu_count
from collections import defaultdict, OrderedDict

from georesolver.prober import RIPEAtlasProber
from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.clickhouse import (
    Query,
    InsertCSV,
    CreateIPv6VPsTable,
    GetHostnamesAnswerSubnet,
)
from georesolver.clickhouse.queries import (
    load_vps,
    get_tables,
    get_targets,
    load_targets,
    get_hostnames,
    get_min_rtt_per_vp,
    get_measurement_ids,
    get_pings_per_src_dst,
    get_mapping_per_hostname,
    get_pings_per_target_extended,
)
from georesolver.prober.ripe_api import RIPEAtlasAPI
from georesolver.agent.insert_process import retrieve_pings, retrieve_traceroutes
from georesolver.evaluation.evaluation_hostname_functions import (
    parse_name_servers,
    get_all_name_servers,
    get_hostname_selection,
    get_hostname_per_main_org,
    get_best_hostnames_per_org,
    parse_hostname_per_main_org,
    get_hostname_per_org_per_ns,
    get_hostname_per_name_server,
    get_hostnames_org_and_ns_threshold,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    get_proportion_under,
    plot_multiple_cdf,
)
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_d_errors_ref,
    get_d_errors_georesolver,
    get_vp_selection_per_target,
)
from georesolver.common.geoloc import (
    haversine,
    greedy_selection_probes_impl,
    filter_default_country_geolocation,
    compute_remove_wrongly_geolocated_probes,
)
from georesolver.common.ip_addresses_utils import (
    get_host_ip_addr,
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.files_utils import (
    load_csv,
    dump_csv,
    load_json,
    dump_json,
    load_anycatch_data,
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

RESULTS_PATH: Path = path_settings.RESULTS_PATH / "ipv6"


def merge_main_orgs(main_org: str) -> None:
    """merge main organization (ex: AMAZON-02 and AMAZON-AES) -> AMAZON"""
    if "AMAZON" in main_org:
        main_org = "AMAZON"

    elif "GOOGLE" in main_org:
        main_org = "GOOGLE"

    elif "AKAMAI" in main_org:
        main_org = "AKAMAI"

    elif "APPLE" in main_org:
        main_org = "APPLE"

    elif "MICROSOFT" in main_org:
        main_org = "MICROSOFT"

    elif "TENCENT" in main_org:
        main_org = "TENCENT"

    elif "CHINANET" in main_org:
        main_org = "CHINANET"

    elif "CMNET" in main_org:
        main_org = "CMNET"

    else:
        main_org = main_org

    return main_org


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
        vps = load_vps(input_table, ipv6=True)

    return vps


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
        logger.warning(f"{hostname_ecs_table=} already exists, skipping ECS filtering")


def name_servers_resolution(
    hostname_ns_table: str, hostname_ecs_table: str
) -> list[str]:
    """perform NS resolution on a set of hostnames"""
    tables = get_tables()
    if hostname_ns_table not in tables:
        host_addr = get_host_ip_addr(ipv6=True)
        host_subnet = get_prefix_from_ip(host_addr, ipv6=True)
        hostnames = get_hostnames(hostname_ecs_table)
        ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ecs_hostnames_ipv6.csv"
        dump_csv(hostnames, ecs_hostnames_path)

        asyncio.run(
            run_dns_mapping(
                subnets=[host_subnet],
                hostname_file=ecs_hostnames_path,
                request_type="NS",
                output_table=hostname_ns_table,
                ipv6=True,
            )
        )
    else:
        logger.warning(f"{hostname_ns_table=} already exists, skipping ECS filtering")


def get_hostname_cdn(hostname_ecs_table: str, output_path: Path) -> None:
    """for each IP address returned by a hostname, retrieve the CDN behind"""

    if output_path.exists():
        return load_json(output_path)

    asndb = pyasn(str(path_settings.STATIC_FILES / "rib_table_v6.dat"))

    asn_to_org = {}
    with (path_settings.STATIC_FILES / "20240101.as-org2info.jsonl").open("r") as f:
        for row in f.readlines():
            row = json.loads(row)
            if "asn" in row and "name" in row:
                asn_to_org[int(row["asn"])] = row["name"]

    with ClickHouseClient(**ch_settings.clickhouse) as client:
        resp = GetHostnamesAnswerSubnet().execute_iter(
            client=client,
            table_name=hostname_ecs_table,
            hostname_filter="",
        )

        org_per_hostname = defaultdict(dict)
        for row in resp:
            hostname = row["hostname"]
            answers = row["answers"]

            for answer in answers:
                asn, bgp_prefix = route_view_bgp_prefix(answer, asndb)

                if not asn or not bgp_prefix:
                    continue

                if "/" not in bgp_prefix:
                    logger.error("Invalid bgp prefix")
                    continue

                if asn:
                    try:
                        org = asn_to_org[asn]
                        try:
                            org_per_hostname[hostname][org].append(bgp_prefix)
                        except KeyError:
                            org_per_hostname[hostname][org] = [bgp_prefix]
                    except KeyError:
                        continue

        for hostname in org_per_hostname:
            for org, bgp_prefixes in org_per_hostname[hostname].items():
                org_per_hostname[hostname][org] = list(set(bgp_prefixes))

        dump_json(
            org_per_hostname,
            output_path,
        )


def filter_anycast_hostnames(input_table: str) -> None:
    """get all answers, compare with anycatch database"""
    asndb = pyasn(str(path_settings.STATIC_FILES / "rib_table_v6.dat"))
    anycatch_db = load_anycatch_data()
    with ClickHouseClient(**ch_settings.clickhouse) as client:
        ecs_hostnames = GetHostnamesAnswerSubnet().execute(
            client=client, table_name=input_table
        )

    anycast_hostnames = set()
    filtered_hostnames = set()
    for row in ecs_hostnames:
        hostname = row["hostname"]
        answer_subnets = row["answers"]

        for s in answer_subnets:
            bgp_prefix = route_view_bgp_prefix(s, asndb)

        if bgp_prefix in anycatch_db:
            anycast_hostnames.add(hostname)
            continue

        filtered_hostnames.add(hostname)

    logger.info(f"Number of unicast hostnames:: {len(filtered_hostnames)}")

    return filtered_hostnames


def filter_most_represented_org(
    hostname_ecs_table: str, input_path: Path, output_path: Path
) -> list[str]:
    """
    from a dict of hostnames and their hosting infrastructure,
    remove some to obtain a tracktable list of hostnames
    on which to perfrom RIPE Atlas probes /24 subnets ECS mapping
    remove anycast prefixes
    """
    anycast_filtered_hostnames = filter_anycast_hostnames(hostname_ecs_table)
    if output_path.exists():
        return load_csv(output_path)

    org_per_hostname = load_json(input_path)

    hostname_per_org = defaultdict(set)
    for hostname, prefix_per_org in org_per_hostname.items():
        # do not consider anycast hostnames
        if hostname not in anycast_filtered_hostnames:
            continue

        for org, _ in prefix_per_org.items():
            org = merge_main_orgs(org)
            hostname_per_org[org].add(hostname)

    hostnae_per_org = OrderedDict(
        sorted(hostname_per_org.items(), key=lambda x: len(x[-1]), reverse=True)
    )

    max_hostnames_per_org = 1_500
    hostnames_to_resolve = set()
    for org, hostnames in hostname_per_org.items():
        if len(hostnames) > max_hostnames_per_org:
            hostnames = sample(list(hostnames), max_hostnames_per_org)
            hostnames_to_resolve.update(hostnames)
        else:
            hostnames_to_resolve.update(hostnames)

    logger.info(f"Number of hostnames to resolver:: {len(hostnames_to_resolve)}")

    dump_csv(hostnames_to_resolve, output_path)

    return hostnames_to_resolve


def get_bpg_prefix_per_hostnames(cdn_per_hostname: dict) -> dict:
    """get all bgp prefixes per hostnames,
    filter hostnames for which we only have one bgp prefix
    """
    bgp_prefix_per_hostname = defaultdict(set)
    for hostname, bgp_prefixes_per_cdn in cdn_per_hostname.items():
        for bgp_prefixes in bgp_prefixes_per_cdn.values():
            bgp_prefix_per_hostname[hostname].update(bgp_prefixes)

    count = 0
    count_1 = 0
    for hostname, bgp_prefixes in bgp_prefix_per_hostname.items():
        if len(bgp_prefixes) < 10:
            count += 1
        if not len(bgp_prefixes) > 1:
            count_1 += 1

    logger.info(
        f"Removed:: {count}/{len(bgp_prefix_per_hostname)}/{round(count * 100 / len(bgp_prefix_per_hostname),2)} hostname with less than 10"
    )

    logger.info(
        f"Removed:: {count_1}/{len(bgp_prefix_per_hostname)}/{round(count_1 * 100 / len(bgp_prefix_per_hostname),2)} hostname with less than 1"
    )

    return bgp_prefix_per_hostname


def get_greedy_vp_selection(output_file: Path) -> None:
    """select VPs with greedy selection to cover maximum coverage"""
    logger.info("Starting greedy algorithm")

    if not output_file.exists():
        selected_probes = []
        vp_distance_matrix = load_json(
            path_settings.DATASET / "vps_pairwise_distance_ipv6.json"
        )
        remaining_probes = set(vp_distance_matrix.keys())
        usable_cpu = cpu_count() - 1
        with Pool(usable_cpu) as p:
            while len(remaining_probes) > 0 and len(selected_probes) < 1_000:
                args = []
                for probe in remaining_probes:
                    args.append((probe, vp_distance_matrix[probe], selected_probes))

                results = p.starmap(greedy_selection_probes_impl, args)

                furthest_probe_from_selected, _ = max(results, key=lambda x: x[1])
                selected_probes.append(furthest_probe_from_selected)
                remaining_probes.remove(furthest_probe_from_selected)

        dump_json(selected_probes, output_file)
        return selected_probes
    else:
        logger.warning("VP greedy selection done, skipping step")
        return load_json(output_file)


def get_hostname_geo_score(vps_per_id: dict, dns_table: str, nb_vps: int) -> dict:
    """
    get all vps mapping, take N vps that maximize geo distance
    compute redirection similarity over the N vps
    if high similarity, hostname does not provide usefull geo info
    """
    # get hostnames geo score (inverse of georesolver)
    hostname_geo_score_path = (
        path_settings.RESULTS_PATH / "hostname_geo_score_ipv6.json"
    )
    if (hostname_geo_score_path).exists():
        return load_json(hostname_geo_score_path)

    greedy_vps = get_greedy_vp_selection(path_settings.DATASET / "greedy_vps_ipv6.json")
    greedy_vps = [vps_per_id[int(id)]["subnet"] for id in greedy_vps[:nb_vps]]

    mapping_per_hostname = get_mapping_per_hostname(
        dns_table=dns_table,
        subnets=[s for s in greedy_vps],
        ipv6=True,
    )
    hostname_similarity_score = {}
    for hostname, vps_mapping in tqdm(mapping_per_hostname.items()):
        pairwise_similarity = []
        for i, vp_i in enumerate(greedy_vps):
            try:
                mapping_i = vps_mapping[vp_i]
            except KeyError:
                continue

            for j, vp_j in enumerate(greedy_vps):
                if vp_i == vp_j:
                    continue

                try:
                    mapping_j = vps_mapping[vp_j]
                except KeyError:
                    continue

                jaccard_index = len(set(mapping_i).intersection(set(mapping_j))) / len(
                    set(mapping_i).union(set(mapping_j))
                )

                pairwise_similarity.append(jaccard_index)

        # take average pairwise similarity
        if pairwise_similarity:
            hostname_similarity_score[hostname] = np.mean(pairwise_similarity)

    hostname_similarity_score = sorted(
        hostname_similarity_score.items(), key=lambda x: x[-1]
    )

    dump_json(hostname_similarity_score, hostname_geo_score_path)

    return hostname_similarity_score


def select_hostnames(vps: list[dict]) -> None:
    """select hostnames based on name servers info, hosting information and returned BGP prefixes"""

    if not (
        path_settings.HOSTNAME_FILES / "ipv6_hostnames_org_bgp_prefixes.json"
    ).exists():
        get_hostname_cdn(
            "vps_ecs_mapping_ipv6_filtered_hostnames",
            path_settings.HOSTNAME_FILES / "ipv6_hostnames_org_bgp_prefixes.json",
        )

    cdn_per_hostname = load_json(
        path_settings.HOSTNAME_FILES / "ipv6_hostnames_org_bgp_prefixes.json"
    )
    bgp_prefix_per_hostnames = get_bpg_prefix_per_hostnames(cdn_per_hostname)
    name_servers_per_hostname = get_all_name_servers("ns_hostnames_ipv6")
    # get geo-scores
    vps_per_id = {}
    for vp in vps:
        vps_per_id[vp["id"]] = vp

    hostname_geo_score = get_hostname_geo_score(
        vps_per_id,
        "vps_ecs_mapping_ipv6_filtered_hostnames",
        1_000,
    )

    # select hostnames based on their name server and organisation
    tlds = load_csv(path_settings.DATASET / "tlds.csv")
    tlds = [t.lower() for t in tlds]
    selected_hostnames, ns_per_hostname = select_hostname_per_org_per_ns(
        name_servers_per_hostname=name_servers_per_hostname,
        tlds=tlds,
        bgp_per_cdn_per_hostname=cdn_per_hostname,
        bgp_prefixes_per_hostname=bgp_prefix_per_hostnames,
        main_org_threshold=0.01,
        bgp_prefix_threshold=2,
    )

    # save files
    _, best_hostnames_per_org_per_ns = get_best_hostnames_per_org(
        hostname_geo_score=hostname_geo_score,
        cdn_per_hostname=cdn_per_hostname,
        bgp_prefix_per_hostnames=bgp_prefix_per_hostnames,
        ns_per_hostname=ns_per_hostname,
        hostnames_per_org_path=(
            path_settings.DATASET / "best_hostnames_per_org_ipv6.json"
        ),
        hostnames_per_org_per_ns_path=(
            path_settings.DATASET / "best_hostnames_per_org_per_ns_ipv6.json"
        ),
    )

    get_hostnames_org_and_ns_threshold(
        best_hostnames_per_org_per_ns, "hostname_georesolver_ipv6_"
    )


def select_hostname_per_org_per_ns(
    name_servers_per_hostname: dict,
    tlds: list[str],
    bgp_per_cdn_per_hostname: dict,
    bgp_prefixes_per_hostname: dict,
    main_org_threshold: float = 0.2,
    bgp_prefix_threshold: int = 2,
) -> dict:
    """perform hostname selection per name server and organization"""
    name_servers_per_hostname = parse_name_servers(name_servers_per_hostname, tlds)

    hostname_per_name_servers = get_hostname_per_name_server(name_servers_per_hostname)

    hostname_per_main_org = get_hostname_per_main_org(
        bgp_per_cdn_per_hostname, main_org_threshold
    )

    hostname_per_main_org = parse_hostname_per_main_org(
        hostname_per_main_org, bgp_prefixes_per_hostname, bgp_prefix_threshold
    )

    main_org_per_hostname = {}
    for org, hostnames in hostname_per_main_org.items():
        for hostname in hostnames:
            main_org_per_hostname[hostname] = org

    hostname_per_org_per_name_servers = get_hostname_per_org_per_ns(
        hostname_per_name_servers, main_org_per_hostname, bgp_prefixes_per_hostname
    )
    selected_hostnames, name_server_per_hostnames = get_hostname_selection(
        hostname_per_org_per_name_servers
    )

    return selected_hostnames, name_server_per_hostnames


def select_ecs_hostnames(
    vps_ecs_mapping_table: str,
    hostname_ecs_table: str,
    vps_table: str,
) -> list[str]:
    """select a subset of ECS-DNS hostnames for calculating DNS resolution similarities"""
    tables = get_tables()
    if vps_ecs_mapping_table not in tables:
        vps = load_vps(vps_table, ipv6=True)
        vps_subnet = list(set([vp["subnet"] for vp in vps]))
        ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ecs_hostnames_ipv6.csv"

        if not ecs_hostnames_path.exists():
            # select best IPv4 hostnames
            selected_hostnames = set()
            hostnames_per_ns_per_org = load_json(
                path_settings.HOSTNAME_FILES / "hostname__5_BGP_10_org_ns.json"
            )
            for _, hostnames_per_org in hostnames_per_ns_per_org.items():
                for _, hostnames in hostnames_per_org.items():
                    selected_hostnames.update(hostnames)

            logger.info(f"IPv4 Selected hostnames:: {len(selected_hostnames)}")

            ipv6_ecs_hostnames = get_hostnames(hostname_ecs_table)
            selected_hostnames = set(ipv6_ecs_hostnames).intersection(
                selected_hostnames
            )

            logger.info(f"IPv6 Selected hostnames:: {len(selected_hostnames)}")

            dump_csv(selected_hostnames, ecs_hostnames_path)

        asyncio.run(
            run_dns_mapping(
                subnets=vps_subnet,
                hostname_file=ecs_hostnames_path,
                request_type="AAAA",
                output_table=vps_ecs_mapping_table,
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


def get_vp_distance_matrix(vp_coordinates: dict, vps: list[dict]) -> dict:
    """compute pairwise distance between each VPs and a list of target ids (RIPE Atlas anchors)"""
    logger.info(f"Calculating VP distance matrix for {len(vp_coordinates)}x{len(vps)}")

    vp_distance_matrix = defaultdict(dict)
    # for better efficiency only compute pairwise distance for the targets
    for vp in tqdm(vps):
        vp_i_id = str(vp["id"])
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
        vp_distance_matrix = get_vp_distance_matrix(vp_coordinates, vps)
        dump_json(vp_distance_matrix, path_settings.VPS_PAIRWISE_DISTANCE_IPv6)

    if not (path_settings.DATASET / "wrongly_geolocated_probes_ipv6.json").exists():
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

    removed_probes = load_json(
        path_settings.DATASET / "wrongly_geolocated_probes_ipv6.json"
    )

    return removed_probes


def filter_vps(vps_table: str, pings_table: str, output_vps_table: str) -> None:
    """filter VPs based on default geoloc and SOI violatio"""
    targets = load_targets(vps_table, ipv6=True)
    vps = load_vps(vps_table, ipv6=True)

    ping_filtered = filter_wrongful_geolocation(targets, vps, pings_table)
    country_filtered = filter_default_country_geolocation(
        vps, path_settings.DATASET / "country_filtered_vps_ipv6.json"
    )

    ping_filtered = [tuple(l) for l in ping_filtered]
    removed_vps = list(set(ping_filtered).union(set(country_filtered)))

    dump_json(removed_vps, path_settings.DATASET / "removed_vps_ipv6.json")

    logger.info(f"{len(removed_vps)} VPs removed")

    filtered_ids = [id for id, _ in removed_vps]

    csv_data = []
    for vp in vps:
        if str(vp["id"]) in filtered_ids:
            continue

        row = []
        for val in vp.values():
            row.append(f"{val}")

        row = ",".join(row)
        csv_data.append(row)

    # create filtered table
    with ClickHouseClient(**ch_settings.clickhouse) as client:
        tmp_file_path = create_tmp_csv_file(csv_data)

        CreateIPv6VPsTable().execute(
            client=client,
            table_name=output_vps_table,
        )

        Query().execute_insert(
            client=client,
            table_name=output_vps_table,
            in_file=tmp_file_path,
        )

        tmp_file_path.unlink()

    return removed_vps


def filter_low_connectivity_vps(
    vps_table: str,
    traceroute_table: str,
    output_vps_table: str,
    delay_threshold: int = 2,
) -> None:
    """filter all VPs for which the measured minimum last mile delay above 2ms"""
    vps = load_vps(vps_table, ipv6=True)
    last_mile_delay = get_min_rtt_per_vp(traceroute_table)

    logger.info(f"Original number of VPs:: {len(vps)}")
    logger.info(f"NB vps traceroutes:: {len(last_mile_delay)}")

    # only keep vps with a last mile delay under threshold
    filtered_vps = []
    for vp in vps:
        try:
            min_rtt = last_mile_delay[vp["addr"]]
            if min_rtt <= delay_threshold:
                filtered_vps.append((vp["id"], vp["addr"]))
        except KeyError:
            continue

    logger.info(f"Remaining VPs after last mile delay filtering:: {len(filtered_vps)}")

    filtered_vps_ids = [vp_id for vp_id, _ in filtered_vps]

    csv_data = []
    for vp in vps:
        if vp["id"] not in filtered_vps_ids:
            continue

        row = []
        for val in vp.values():
            row.append(f"{val}")

        row = ",".join(row)
        csv_data.append(row)

    with ClickHouseClient(**ch_settings.clickhouse) as client:
        tmp_file_path = create_tmp_csv_file(csv_data)

        CreateIPv6VPsTable().execute(client=client, table_name=output_vps_table)

        Query().execute_insert(
            client=client, table_name=output_vps_table, in_file=tmp_file_path
        )

        tmp_file_path.unlink()


def meshed_traceroutes_schedule(
    vps_table: str,
    max_nb_traceroutes: int = 100,
    input_file: Path = None,
    check_cache: bool = True,
) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    if input_file:

        measurement_schedule = load_json(input_file)
        logger.info(f"Schedule for {len(measurement_schedule)} targets")
        # todo: filter
        return measurement_schedule

    vps = load_vps(vps_table, ipv6=True)
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
    for vp in tqdm(vps):
        closest_vp_ids = ordered_distance_matrix[str(vp["id"])]

        closest_vp_addrs = []
        for id, _ in closest_vp_ids:
            try:
                vp = vp_per_id[id]
                if IPv6Address(vp["addr"]).is_private:
                    continue
                closest_vp_addrs.append(vp["addr"])
            except KeyError as e:
                continue

        # only take targets outside of VPs subnet for meshed traceroutes
        i = 0
        for addr in closest_vp_addrs:
            if get_prefix_from_ip(addr, ipv6=True) != vp["subnet"]:
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
        # prober.main(measurement_schedule),
        insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", measurement_tag],
            output_table=output_table,
            measurement_type=measurement_type,
        ),
    )

    logger.info("Meshed CDNs pings measurement done")


def evaluation(mapping_table: str, hostname_file: Path) -> None:

    cdfs = []
    removed_vps = load_json(path_settings.DATASET / "removed_vps_ipv6.json")
    hostnames = load_csv(hostname_file)
    targets = load_targets(ch_settings.VPS_FILTERED_TABLE + "_ipv6", ipv6=True)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE + "_ipv6", ipv6=True)
    vps_coordinates = {vp["addr"]: vp for vp in vps}
    target_subnets = [t["subnet"] for t in targets]
    vp_subnets = [v["subnet"] for v in vps]

    tables = get_tables()
    if mapping_table not in tables:
        asyncio.run(
            run_dns_mapping(
                subnets=[v["subnet"] for v in vps],
                hostname_file=hostname_file,
                output_table=mapping_table,
                name_servers="8.8.8.8",
                request_type="AAAA",
                ipv6=True,
            )
        )

    # load score similarity between vps and targets
    scores = get_scores(
        output_path=RESULTS_PATH / "score.pickle",
        hostnames=hostnames,
        target_subnets=target_subnets,
        vp_subnets=vp_subnets,
        target_ecs_table=mapping_table,
        vps_ecs_table=mapping_table,
        ipv6=True,
    )

    vp_selection_per_target = get_vp_selection_per_target(
        output_path=RESULTS_PATH / "vp_selection.pickle",
        scores=scores,
        targets=[t["addr"] for t in targets],
        vps=vps,
        ipv6=True,
    )

    pings_per_target = get_pings_per_target_extended(
        ch_settings.VPS_MESHED_PINGS_TABLE + "_ipv6", removed_vps, ipv6=True
    )

    # add reference
    d_errors_ref = get_d_errors_ref(pings_per_target, vps_coordinates)
    x, y = ecdf(d_errors_ref)
    cdfs.append((x, y, "Shortest ping, all VPs"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"Shortest Ping all VPs:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"Shortest Ping all VPs::: median_error={round(m_error, 2)} [km]")

    # add iterative resolver results
    d_errors = get_d_errors_georesolver(
        targets=[t["addr"] for t in targets],
        pings_per_target=pings_per_target,
        vp_selection_per_target=vp_selection_per_target,
        vps_coordinates=vps_coordinates,
        probing_budget=50,
    )
    x, y = ecdf(d_errors)
    cdfs.append((x, y, "GeoResolver"))

    m_error = round(np.median(x), 2)
    proportion_of_ip = get_proportion_under(x, y)

    logger.info(f"GeoResolver:: <40km={round(proportion_of_ip, 2)}")
    logger.info(f"GeoResolver:: median_error={round(m_error, 2)} [km]")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="ipv6_ripe_atlas",
        metric_evaluated="d_error",
        legend_pos="lower right",
    )


def ipV6_sample() -> None:
    """run IPv6 georesolver on a sample of IPv6 addrs"""
    # 1. IF evaluation conclusive, run sample IPv6
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
    do_meshed_pings: bool = False
    do_meshed_traceroutes: bool = False
    do_evaluation: bool = True

    # 1. get vps
    vps = asyncio.run(get_vps(ch_settings.VPS_RAW_TABLE + "_ipv6"))

    if do_select_hostnames:
        filter_ecs_hostnames(hostname_ecs_table="ecs_hostnames_ipv6")
        name_servers_resolution(
            hostname_ns_table="ns_hostnames_ipv6",
            hostname_ecs_table="ecs_hostnames_ipv6",
        )
        get_hostname_cdn(
            "ecs_hostnames_ipv6",
            path_settings.HOSTNAME_FILES / "ecs_hostnames_organization_ipv6.json",
        )
        filter_most_represented_org(
            hostname_ecs_table="ecs_hostnames_ipv6",
            input_path=path_settings.HOSTNAME_FILES
            / "ecs_hostnames_organization_ipv6.json",
            output_path=path_settings.HOSTNAME_FILES
            / "hostnames_ecs_filtered_most_represented_org_ipv6.csv",
        )

        # perform ecs resolution over all filtered hostnames/ and RIPE Atlas probes /24
        tables = get_tables()
        output_table = "vps_ecs_mapping_ipv6_filtered_hostnames"
        if not output_table in tables:
            asyncio.run(
                run_dns_mapping(
                    subnets=[vp["subnet"] for vp in vps],
                    output_table=output_table,
                    request_type="AAAA",
                    name_servers="8.8.8.8",
                    ipv6=True,
                    hostname_file=path_settings.HOSTNAME_FILES
                    / "hostnames_ecs_filtered_most_represented_org_ipv6.csv",
                )
            )
        else:
            logger.warning(
                f"Skipping ECS resolution because table: {output_table=} already exists"
            )

        # select hostnames
        select_hostnames(vps)

    if do_meshed_pings:
        measurement_schedule = meshed_pings_schedule(
            ch_settings.VPS_RAW_TABLE + "_ipv6"
        )
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
        filter_vps(
            ch_settings.VPS_RAW_TABLE + "_ipv6",
            ch_settings.VPS_MESHED_PINGS_TABLE + "_ipv6",
            ch_settings.VPS_FILTERED_TABLE + "_ipv6",
        )

        # remove if no previous schedule was calculated
        schedule_path = (
            path_settings.MEASUREMENTS_SCHEDULE
            / "vps_meshed_traceroutes_ipv6__92c5722e-0a9f-43c9-983b-e3e2e5bb2bb0.json"
        )
        measurement_schedule = meshed_traceroutes_schedule(
            ch_settings.VPS_FILTERED_TABLE + "_ipv6", input_file=schedule_path
        )

        asyncio.run(
            run_measurement(
                measurement_schedule=measurement_schedule,
                measurement_tag="meshed-traceroutes-ipv6",
                output_table=ch_settings.VPS_MESHED_TRACEROUTE_TABLE + "_ipv6",
                measurement_type="traceroute",
                check_cache=True,
            )
        )

        # filter low connectivity vps
        filter_low_connectivity_vps(
            vps_table=ch_settings.VPS_FILTERED_TABLE + "_ipv6",
            traceroute_table=ch_settings.VPS_MESHED_TRACEROUTE_TABLE + "_ipv6",
            output_vps_table=ch_settings.VPS_FILTERED_FINAL_TABLE + "_ipv6",
        )

    if do_evaluation:
        evaluation(
            mapping_table="vps_ecs_mapping_ecs_ipv6_latest",
            hostname_file=path_settings.HOSTNAME_FILES
            / "hostname_georesolver_ipv6__10_BGP_5_org_ns_new.csv",
        )


if __name__ == "__main__":
    main()
