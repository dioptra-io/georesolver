"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio
import time

from datetime import timedelta, datetime
from pyasn import pyasn
from tqdm import tqdm
from pathlib import Path
from numpy import mean
from loguru import logger
from pych_client import AsyncClickHouseClient
from collections import defaultdict

from geogiant.zdns import ZDNS
from geogiant.clickhouse import (
    GetVPsSubnets,
    GetSubnets,
    GetSubnetPerHostname,
    GetVPSInfoPerSubnet,
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
    load_csv,
    dump_json,
    load_json,
)
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def resolve_hostnames(
    subnets: list,
    hostname_file: Path,
    iterative: bool = False,
    repeat: bool = False,
    chunk_size: int = 100,
    output_file: Path = None,
    output_table: str = None,
    end_date: datetime = "2024-01-23 00:00",
    waiting_time: int = 60 * 60 * 2,
    request_timout: float = 0.1,
    request_type: str = "A",
) -> None:
    """repeat zdns measurement on set of VPs"""

    if not output_file and not output_table:
        raise RuntimeError("Either output_file or output_table var must be set")

    async def raw_dns_mapping(subnets: list) -> None:
        """perform DNS mapping with zdns on VPs subnet"""

        subnets = [subnet + "/24" for subnet in subnets]

        with hostname_file.open("r") as f:
            logger.info("raw hostname file already generated")

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
                )
                await zdns.main()

                tmp_hostname_file.unlink()

    await raw_dns_mapping(subnets)

    if repeat:
        time.sleep(waiting_time)  # wait two hours
        while datetime.today() < end_date:
            await raw_dns_mapping(subnets)
            time.sleep(waiting_time)


def get_geographic_mapping(subnets: dict, vps_per_subnet: dict) -> tuple[list, list]:
    """get vps country and continent for a given hostname bgp prefix"""
    mapping_countries = []
    for subnet in subnets:
        for vp in vps_per_subnet[subnet]:
            country_code = vp[-1]
            mapping_countries.append(country_code)

    return mapping_countries


def get_geographic_ratio(geographic_mapping: list) -> float:
    """return the ratio of the most represented geographic granularity (country/continent)"""
    try:
        return max(
            [
                geographic_mapping.count(region) / len(geographic_mapping)
                for region in geographic_mapping
            ]
        )
    except ValueError:
        return 0


def filter_on_geo_distribution(
    bgp_prefix_country_ratio: list[float],
    hostnames_geo_mapping: dict[list],
    threshold_country: float = 0.7,
) -> dict:
    """
    get hostnames and their major geographical region of influence.
    Return True if the average ratio for each hostname's BGP prefix is above
    a given threshold (80% of VPs in the same region by default)
    """
    valid_hostnames = []
    invalid_hostnames = []
    for country_ratio in hostnames_geo_mapping.items():
        avg_ratio_country = []

        for major_country_ratio in bgp_prefix_country_ratio:
            avg_ratio_country.append(major_country_ratio)

        avg_ratio_country = mean(avg_ratio_country)
        if avg_ratio_country > threshold_country:
            valid_hostnames.append((avg_ratio_country))
        else:
            invalid_hostnames.append((avg_ratio_country))

    return valid_hostnames, invalid_hostnames


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

    vps_subnet = [get_prefix_from_ip(get_host_ip_addr())]

    if not vps_subnet:
        raise RuntimeError(
            f"Either var input_file or input_table must be set to load vps subnets"
        )

    if not output_file and not output_table:
        raise RuntimeError(
            f"Either var output_file or output_table must be set to dump results"
        )

    # output file if out file instead of output table
    await resolve_hostnames(
        subnets=vps_subnet,
        hostname_file=tmp_hostname_file,
        output_file=output_file,
        output_table=output_table,
        repeat=False,
        end_date=None,
        chunk_size=100,
        request_type="NS",
    )

    tmp_hostname_file.unlink()


async def resolve_vps_subnet(
    selected_hostnames: list,
    input_file: Path = None,
    output_file: Path = None,
    input_table: str = None,
    output_table: str = None,
) -> None:
    """perform ECS-DNS resolution one all VPs subnet"""
    tmp_hostname_file = create_tmp_csv_file(selected_hostnames)

    # load subnets from file if in file
    if input_file:
        vps_subnet = load_json(input_file)

    if input_table:
        async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
            vps_subnet = await GetSubnets().aio_execute(
                client=client, table_name=input_table
            )

    if not vps_subnet:
        raise RuntimeError(
            f"Either var input_file or input_table must be set to load vps subnets"
        )

    if not output_file and not output_table:
        raise RuntimeError(
            f"Either var output_file or output_table must be set to dump results"
        )

    # output file if out file instead of output table
    await resolve_hostnames(
        subnets=[row["subnet"] for row in vps_subnet],
        hostname_file=tmp_hostname_file,
        output_file=output_file,
        output_table=output_table,
        repeat=False,
        end_date=None,
        chunk_size=100,
    )

    tmp_hostname_file.unlink()


async def get_hostname_cdn(input_table: str) -> None:
    """for each IP address returned by a hostname, retrieve the CDN behind"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

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
            org_per_hostname, path_settings.DATASET / "ecs_hostnames_organization.json"
        )


async def resolve_vps_on_selected_hostnames(
    selected_hostnames_file: Path, output_table: str, repeat_for: int = 7
) -> None:
    """
    perform ECS-DNS resolution one all VPs subnet
    repeat_for: number of days to repeat the experiment
    """
    end_date = datetime.today() + timedelta(days=repeat_for)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps_subnet = await GetVPsSubnets().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )

    await resolve_hostnames(
        subnets=[row["subnet"] for row in vps_subnet],
        hostname_file=selected_hostnames_file,
        output_table=output_table,
        repeat=True,
        end_date=end_date,
        chunk_size=10_000,
        waiting_time=0,
    )


async def main() -> None:
    """init main"""
    logger.info("Starting Geolocation hostnames initialization")

    # await filter_ecs_hostnames(output_table="hostnames_1M_resolution")

    logger.info("Retrieved ECS hostnames")

    logger.info("Get ECS hostnames CDN/organization")
    # await get_hostname_cdn(input_table="hostnames_1M_resolution")

    selected_hostnames = load_csv(path_settings.DATASET / "ecs_selected_hostnames.csv")

    # input_file = path_settings.DATASET / "vps_subnet.json"
    # output_file = path_settings.RESULTS_PATH / "vps_mapping_ecs_resolution.csv"
    # await resolve_vps_subnet(
    #     selected_hostnames=selected_hostnames,
    #     input_file=input_file,
    #     output_file=output_file,
    # )

    await resolve_name_servers(
        selected_hostnames=selected_hostnames,
        output_file=path_settings.RESULTS_PATH / "name_server_resolution.csv",
    )

    logger.info("Hostname resolution done for every VPs")

    # await filter_geo_hostnames()

    # logger.info("Geographically valid hostname done")

    # await get_hostname_cdn()

    # await resolve_vps_on_selected_hostnames(
    #     selected_hostnames_file=path_settings.DATASET
    #     / "hostname_1M_max_bgp_prefix_per_cdn.csv",
    #     output_table="time_of_day_evaluation",
    # )


if __name__ == "__main__":
    asyncio.run(main())
