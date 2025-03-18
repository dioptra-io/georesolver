"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio

from pyasn import pyasn
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from pych_client import AsyncClickHouseClient

from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.clickhouse import (
    GetDNSMapping,
    GetHostnamesAnswerSubnet,
)
from georesolver.clickhouse.queries import (
    load_vp_subnets,
    change_table_name,
    get_tables,
)
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    get_host_ip_addr,
    route_view_bgp_prefix,
)
from georesolver.common.files_utils import (
    load_anycatch_data,
    create_tmp_csv_file,
    dump_json,
    load_csv,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def filter_ecs_hostnames(output_table: str) -> None:
    """perform zdns resolution on host subnet, remove hostnames with source scope equal to 0"""
    host_addr = get_host_ip_addr()
    host_subnet = get_prefix_from_ip(host_addr)

    await run_dns_mapping(
        subnets=[host_subnet],
        hostname_file=path_settings.HOSTNAMES_MILLIONS,
        output_table=output_table,
    )


async def filter_anycast_hostnames(input_table: str) -> None:
    """get all answers, compare with anycatch database"""
    anycatch_db = load_anycatch_data()
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
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

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
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
                        # logger.error(f"Cannot retrieve org for AS:: {asn}")
                        continue

        for hostname in org_per_hostname:
            for org, bgp_prefixes in org_per_hostname[hostname].items():
                org_per_hostname[hostname][org] = list(set(bgp_prefixes))

        dump_json(
            org_per_hostname,
            path_settings.DATASET / "ecs_hostnames_organization_test.json",
        )


async def main() -> None:
    """init main"""
    # TODO: INIT HOSTNAMES
    input_file = path_settings.DATASET / "vps_subnet.json"

    logger.info("Retrieving ECS hostnames")
    await filter_ecs_hostnames(output_table="hostnames_1M_resolution")

    logger.info("Get hostnames hosting organization")
    await get_hostname_cdn(input_table="hostnames_1M_resolution")

    logger.info(f"ECS resolution on VPs subnets")
    selected_hostnames = load_csv(
        path_settings.DATASET / "internet_scale_hostnames.csv"
    )

    await run_dns_mapping(
        selected_hostnames=selected_hostnames,
        input_file=input_file,
        output_table="vps_mapping_ecs_latest",
    )

    logger.info(f"Resolving name servers")
    await resolve_name_servers(
        selected_hostnames=selected_hostnames,
        output_table="name_servers_end_to_end",
    )

    logger.info(f"Final ECS resolution on VP subnets")
    await run_dns_mapping(
        selected_hostnames_file=path_settings.DATASET
        / "hostname_1M_max_bgp_prefix_per_cdn.csv",
        output_table="time_of_day_evaluation",
    )


async def ecs_init(hostname_file: Path) -> None:
    """update vps ECS mapping"""
    logger.info(
        f"Starting VPs ECS mapping, output table:: {clickhouse_settings.VPS_ECS_MAPPING_TABLE}"
    )

    vps_subnets = load_vp_subnets(clickhouse_settings.VPS_FILTERED_FINAL_TABLE)

    # first change name
    tables_name = get_tables()
    new_table_name = (
        clickhouse_settings.VPS_ECS_MAPPING_TABLE
        + f"__{str(datetime.now()).split(' ')[0].replace('-', '_')}"
    )
    if (
        new_table_name in tables_name
        and clickhouse_settings.VPS_ECS_MAPPING_TABLE in tables_name
    ):
        logger.info("VPs ECS table should not be updated every day, skipping step")
        return
    elif (
        new_table_name in tables_name
        and not clickhouse_settings.VPS_ECS_MAPPING_TABLE in tables_name
    ):
        logger.warning("Prev vps mapping table exists but not current one")
    elif (
        new_table_name not in tables_name
        and not clickhouse_settings.VPS_ECS_MAPPING_TABLE in tables_name
    ):
        logger.info("Runnin VPs ECS mapping for the first time")
    else:
        logger.info("Renewing VPS ECS mapping table")
        change_table_name(clickhouse_settings.VPS_ECS_MAPPING_TABLE, new_table_name)

    # finally run ECS mapping and output to default table
    await run_dns_mapping(
        subnets=vps_subnets,
        hostname_file=hostname_file,
        output_table=clickhouse_settings.VPS_ECS_MAPPING_TABLE,
    )


if __name__ == "__main__":
    asyncio.run(main())
