"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio

from pyasn import pyasn
from pathlib import Path
from loguru import logger
from collections import defaultdict
from datetime import datetime
from pych_client import AsyncClickHouseClient

from geogiant.zdns import ZDNS
from geogiant.clickhouse import (
    GetDNSMapping,
    GetHostnamesAnswerSubnet,
)
from geogiant.common.ip_addresses_utils import (
    get_prefix_from_ip,
    get_host_ip_addr,
    route_view_bgp_prefix,
)
from geogiant.common.files_utils import (
    load_anycatch_data,
    create_tmp_csv_file,
    dump_json,
    load_csv,
)
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def resolve_hostnames(
    subnets: list,
    hostname_file: Path,
    iterative: bool = False,
    repeat: bool = False,
    chunk_size: int = 500,
    output_file: Path = None,
    output_table: str = None,
    end_date: datetime = "2024-01-23 00:00",
    waiting_time: int = 60,
    request_timout: float = 0.1,
    request_type: str = "A",
    output_logs: Path = None,
) -> None:
    """repeat zdns measurement on set of VPs"""

    async def raw_dns_mapping(subnets: list) -> None:
        """perform DNS mapping with zdns on VPs subnet"""

        subnets = [subnet + "/24" for subnet in subnets]

        logger.info(f"Starting ECS resolution, {repeat=}, {end_date=}")

        with hostname_file.open("r") as f:
            all_hostnames = f.readlines()

            logger.info(
                f"Resolving:: {len(all_hostnames)} hostnames on {len(subnets)} subnets"
            )

            # split file
            for index in range(0, len(all_hostnames), chunk_size):
                hostnames = all_hostnames[index : index + chunk_size]

                tmp_hostname_file = create_tmp_csv_file(hostnames)

                logger.info(
                    f"Starting to resolve hostnames {index} to {index + chunk_size} (total={len(all_hostnames)})"
                )

                zdns = ZDNS(
                    subnets=subnets,
                    hostname_file=tmp_hostname_file,
                    output_file=output_file,
                    output_table=output_table,
                    name_servers="8.8.8.8",
                    iterative=iterative,
                    timeout=request_timout,
                    request_type=request_type,
                    output_logs=output_logs,
                )
                await zdns.main()

                tmp_hostname_file.unlink()

    await raw_dns_mapping(subnets)

    if repeat:
        await asyncio.sleep(waiting_time)  # wait two hours
        while datetime.today() < end_date:
            await raw_dns_mapping(subnets)
            await asyncio.sleep(waiting_time)


async def filter_ecs_hostnames(output_table: str) -> None:
    """perform zdns resolution on host subnet, remove hostnames with source scope equal to 0"""
    host_addr = get_host_ip_addr()
    host_subnet = get_prefix_from_ip(host_addr)

    await resolve_hostnames(
        subnets=[host_subnet],
        hostname_file=path_settings.HOSTNAMES_MILLIONS,
        output_table=output_table,
        repeat=False,
        end_date=None,
        chunk_size=100,
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
    await resolve_hostnames(
        subnets=["132.227.123.0"],
        hostname_file=tmp_hostname_file,
        output_file=output_file,
        output_table=output_table,
        repeat=False,
        end_date=None,
        chunk_size=1_000,
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

    await resolve_hostnames(
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
    await resolve_hostnames(
        selected_hostnames_file=path_settings.DATASET
        / "hostname_1M_max_bgp_prefix_per_cdn.csv",
        output_table="time_of_day_evaluation",
    )


if __name__ == "__main__":
    asyncio.run(main())
