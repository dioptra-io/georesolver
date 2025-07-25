"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio

from tqdm import tqdm
from pyasn import pyasn
from pathlib import Path
from loguru import logger
from random import sample
from datetime import datetime
from pych_client import AsyncClickHouseClient
from multiprocessing import Pool, cpu_count
from collections import defaultdict, OrderedDict

from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.clickhouse import (
    GetDNSMapping,
    GetHostnamesAnswerSubnet,
)
from georesolver.common.geoloc import greedy_selection_probes_impl
from georesolver.clickhouse.queries import (
    load_vps,
    get_tables,
    get_hostnames,
    load_vp_subnets,
    change_table_name,
    get_mapping_per_hostname,
)
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
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    get_host_ip_addr,
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
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


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
        ecs_hostnames_path = path_settings.HOSTNAME_FILES / "ecs_hostnames_new.csv"
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


async def filter_anycast_hostnames(input_table: str) -> None:
    """get all answers, compare with anycatch database"""
    anycatch_db = load_anycatch_data()
    async with AsyncClickHouseClient(**ch_settings.clickhouse) as client:
        ecs_hostnames = await GetDNSMapping().aio_execute(
            client=client, table_name=input_table
        )

    anycast_hostnames = set()
    all_hostnames = set()
    for row in ecs_hostnames:
        hostname = row["hostname"]
        bgp_prefix = row["answer_bgp_prefix"]

        all_hostnames.add(hostname)

        if bgp_prefix in anycatch_db:
            anycast_hostnames.add(hostname)
            continue

    logger.info(
        f"Number of unicast hostnames:: {len(anycast_hostnames.symmetric_difference(all_hostnames))}"
    )

    logger.info(f"Number of Anycast hostnames:: {len(anycast_hostnames)}")

    return anycast_hostnames.symmetric_difference(all_hostnames)


async def resolve_name_servers(
    selected_hostnames: list,
    output_file: Path = None,
    output_table: str = None,
) -> None:
    """perform ECS-DNS resolution one all VPs subnet"""
    tmp_hostname_file = create_tmp_csv_file(selected_hostnames)

    # output file if out file instead of output table
    await run_dns_mapping(
        subnets=["132.227.123.0"],
        hostname_file=tmp_hostname_file,
        output_file=output_file,
        output_table=output_table,
        chunk_size=5_00,
        request_type="NS",
        request_timout=1,
    )

    tmp_hostname_file.unlink()


async def get_hostname_cdn(input_table: str) -> None:
    """for each IP address returned by a hostname, retrieve the CDN behind"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    anycast_hostnames = filter_anycast_hostnames(input_table)

    asn_to_org = {}
    with (path_settings.DATASET / "20240101.as-org2info.jsonl").open("r") as f:
        for row in f.readlines():
            row = json.loads(row)
            if "asn" in row and "name" in row:
                asn_to_org[int(row["asn"])] = row["name"]

    async with AsyncClickHouseClient(**ch_settings.clickhouse) as client:
        resp = await GetHostnamesAnswerSubnet().aio_execute_iter(
            client=client,
            table_name=input_table,
            hostname_filter="",
        )

        org_per_hostname = defaultdict(dict)
        async for row in resp:
            hostname = row["hostname"]
            answers = row["answers"]

            if hostname in anycast_hostnames:
                continue

            for answer in answers:
                asn, bgp_prefix = route_view_bgp_prefix(answer, asndb)

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
            path_settings.DATASET / "ecs_hostnames_organization_test.json",
        )


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

    hostname_per_org = OrderedDict(
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
            path_settings.DATASET / "vps_pairwise_distance_new.json"
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
    hostname_geo_score_path = path_settings.RESULTS_PATH / "hostname_geo_score_new.json"
    if (hostname_geo_score_path).exists():
        return load_json(hostname_geo_score_path)

    greedy_vps = get_greedy_vp_selection(path_settings.DATASET / "greedy_vps_new.json")
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


def select_hostnames(vps: list[dict]) -> None:
    """select hostnames based on name servers info, hosting information and returned BGP prefixes"""

    if not (
        path_settings.HOSTNAME_FILES / "ipv6_hostnames_org_bgp_prefixes.json"
    ).exists():
        get_hostname_cdn(
            "vps_ecs_mapping_new_filtered_hostnames",
            path_settings.HOSTNAME_FILES / "ipv6_hostnames_org_bgp_prefixes.json",
        )

    cdn_per_hostname = load_json(
        path_settings.HOSTNAME_FILES / "ipv6_hostnames_org_bgp_prefixes.json"
    )
    bgp_prefix_per_hostnames = get_bpg_prefix_per_hostnames(cdn_per_hostname)
    name_servers_per_hostname = get_all_name_servers("ns_hostnames_new")
    # get geo-scores
    vps_per_id = {}
    for vp in vps:
        vps_per_id[vp["id"]] = vp

    hostname_geo_score = get_hostname_geo_score(
        vps_per_id,
        "vps_ecs_mapping_new_filtered_hostnames",
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
            path_settings.DATASET / "best_hostnames_per_org_new.json"
        ),
        hostnames_per_org_per_ns_path=(
            path_settings.DATASET / "best_hostnames_per_org_per_ns_new.json"
        ),
    )

    get_hostnames_org_and_ns_threshold(
        best_hostnames_per_org_per_ns, "hostname_georesolver_new_"
    )


def main() -> None:
    """
    perform Hostname selection using CrUX to 1M domains as input:
        - run ecs_dns resolution on 1M CrUX table (1 subnet)
        - get name servers for each domain name
        - get hosting organization for each domain name
        - filter some of the most represented organization (preserve diversity)
        - run ECS-DNS resolution on selected hostnames for all VPs' subnets
        - select final set of domains name based on returned answers
    """
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)

    filter_ecs_hostnames(hostname_ecs_table="ecs_hostnames_new")
    name_servers_resolution(
        hostname_ns_table="ns_hostnames_new",
        hostname_ecs_table="ecs_hostnames_new",
    )
    get_hostname_cdn(
        "ecs_hostnames_new",
        path_settings.HOSTNAME_FILES / "ecs_hostnames_organization_new.json",
    )
    filter_most_represented_org(
        hostname_ecs_table="ecs_hostnames_new",
        input_path=path_settings.HOSTNAME_FILES / "ecs_hostnames_organization_new.json",
        output_path=path_settings.HOSTNAME_FILES
        / "hostnames_ecs_filtered_most_represented_org_new.csv",
    )

    # perform ecs resolution over all filtered hostnames/ and RIPE Atlas probes /24
    tables = get_tables()
    output_table = "vps_ecs_mapping_new_filtered_hostnames"
    if not output_table in tables:
        asyncio.run(
            run_dns_mapping(
                subnets=[vp["subnet"] for vp in vps],
                output_table=output_table,
                request_type="AAAA",
                name_servers="8.8.8.8",
                ipv6=True,
                hostname_file=path_settings.HOSTNAME_FILES
                / "hostnames_ecs_filtered_most_represented_org_new.csv",
            )
        )
    else:
        logger.warning(
            f"Skipping ECS resolution because table: {output_table=} already exists"
        )

    # select hostnames
    select_hostnames(vps)


async def ecs_init(
    itterative: bool = False,
    hostname_file: Path = path_settings.HOSTNAMES_GEORESOLVER,
    vps_mapping_table: str = ch_settings.VPS_ECS_MAPPING_TABLE,
) -> None:
    """update vps ECS mapping"""
    logger.info(f"Starting VPs ECS mapping, output table:: {vps_mapping_table}")

    vps_subnets = load_vp_subnets(ch_settings.VPS_FILTERED_FINAL_TABLE)

    # first change name
    tables_name = get_tables()
    new_table_name = (
        vps_mapping_table + f"_{str(datetime.now()).split(' ')[0].replace('-', '_')}"
    )

    if new_table_name in tables_name and vps_mapping_table in tables_name:
        logger.warning("VPs ECS table should not be updated every day, skipping step")
        return
    elif new_table_name in tables_name and not vps_mapping_table in tables_name:
        logger.warning("Prev vps mapping table exists but not current one")
    elif new_table_name not in tables_name and not vps_mapping_table in tables_name:
        logger.info("Running VPs ECS mapping for the first time")
    else:
        logger.info("Renewing VPS ECS mapping table")
        change_table_name(vps_mapping_table, new_table_name)

    # finally run ECS mapping and output to default table
    await run_dns_mapping(
        subnets=vps_subnets,
        hostname_file=hostname_file,
        output_table=vps_mapping_table,
        itterative=itterative,
    )


if __name__ == "__main__":
    main()
