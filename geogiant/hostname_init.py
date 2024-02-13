"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import asyncio
import schedule
import time

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from numpy import mean
from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient
from multiprocessing import Pool, cpu_count


from geogiant.zdns import ZDNS
from geogiant.clickhouse import (
    GetSubnets,
    GetSubnetPerHostname,
    GetVPSInfoPerSubnet,
    GetDNSMapping,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, get_host_ip_addr
from geogiant.common.files_utils import (
    load_countries_info,
    load_anycatch_data,
    dump_csv,
    create_tmp_csv_file,
)
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def resolve_hostnames(
    subnets: list,
    hostname_file: Path,
    output_table: str,
    repeat: bool = False,
    end_date: str = "2024-01-23 00:00",
    chunk_size: int = 1_000,
    max_hostname: int = 100_000,
) -> None:
    """repeat zdns measurement on set of VPs"""

    async def raw_dns_mapping(subnets: list) -> None:
        """perform DNS mapping with zdns on VPs subnet"""

        subnets = [subnet + "/24" for subnet in subnets]

        with hostname_file.open("r") as f:
            logger.info("raw hostname file already generated")

            # take max hostnames from hostname file
            all_hostnames = f.readlines()
            all_hostnames = all_hostnames[:max_hostname]

            # split file
            for index in range(0, len(all_hostnames), chunk_size):
                hostnames = all_hostnames[index : index + chunk_size]

                tmp_hostname_file = create_tmp_csv_file(hostnames)

                logger.info(
                    f"Starting to resolve hostnames {index * chunk_size} to {(index + 1) * chunk_size} (total={len(all_hostnames * chunk_size)})"
                )

                zdns = ZDNS(
                    subnets=subnets,
                    hostname_file=tmp_hostname_file,
                    name_servers="8.8.8.8",
                    table_name=output_table,
                )
                await zdns.main()

                tmp_hostname_file.unlink()
                time.sleep(1)

    await raw_dns_mapping(subnets)

    if repeat:
        # TODO: replace with Crontab module
        schedule.every(4).hours.until(end_date).do(raw_dns_mapping)

        while True:
            schedule.run_pending()


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
    return max(
        [
            geographic_mapping.count(region) / len(geographic_mapping)
            for region in geographic_mapping
        ]
    )


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


async def get_hostnames_geographic_influence() -> tuple[dict, dict]:
    """
    for each hostname, get vps mapped per BGP prefix and get hostname geographical area of influence
    Assumption: hostnames that perform ECS-DNS resolution based on geographical information
    associate BGP prefixes to vps in the same region (either country or continent).
    """
    valid_hostnames = set()
    invalid_hostnames = set()

    batch = 0
    threshold_country = 0.7
    anycatch_prefixes = load_anycatch_data()

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetVPSInfoPerSubnet().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )
        vps_per_subnet = {}
        for row in resp:
            vps_per_subnet[row["subnet"]] = row["vps"]

        subnet_per_hostname = await GetSubnetPerHostname().execute(
            client=client,
            table_name=clickhouse_settings.DNS_MAPPING_VPS_RAW,
            anycast_filter=anycatch_prefixes,
        )

    logger.info(f"Number of hostnames loaded: {len(subnet_per_hostname)}")

    for row in tqdm(subnet_per_hostname):
        hostname = row["hostname"]
        vps_per_bgp_prefix = row["vps_per_bgp_prefix"]

        hostname_country_ratio = []
        for _, subnets in vps_per_bgp_prefix:

            mapping_countries = get_geographic_mapping(subnets, vps_per_subnet)
            major_country_ratio = get_geographic_ratio(mapping_countries)

            hostname_country_ratio.append(major_country_ratio)

        avg_mapping_ratio = mean(hostname_country_ratio)
        if avg_mapping_ratio > threshold_country:
            valid_hostnames.add((f"{hostname}, {avg_mapping_ratio}"))
            logger.debug(
                f"Hostname::{hostname} valid, mapping ratio = {avg_mapping_ratio}"
            )
        else:
            invalid_hostnames.add(f"{hostname}, {avg_mapping_ratio}")
            logger.debug(
                f"Hostname::{hostname} invalid, mapping ratio = {avg_mapping_ratio}"
            )

        batch += 1
        if batch > 10:
            dump_csv(
                valid_hostnames,
                path_settings.DATASET / "valid_hostnames.csv",
            )
            dump_csv(
                invalid_hostnames,
                path_settings.DATASET / "invalid_hostnames.csv",
            )

    return valid_hostnames, invalid_hostnames


def print_hostname_info(
    valid_hostnames: dict[list], invalid_hostnames: dict[list]
) -> None:
    """print info on hostname filtering analysis"""

    logger.info("##########################################################")
    logger.info("# HOSTNAMES GEO VALIDATION: COUNTRY EVAL #################")
    logger.info("##########################################################")
    logger.info(f"Valid hostnames = {len(valid_hostnames['country'])}")
    for hostname, avg_ratio in valid_hostnames["country"]:
        logger.info(f"Hostname = {hostname}, avg ratio = {avg_ratio}")

    logger.info("\n")

    logger.info(f"Invalid hostnames = {len(invalid_hostnames['country'])}")
    for hostname, avg_ratio in invalid_hostnames["country"]:
        logger.info(f"Hostname = {hostname}, avg ratio = {avg_ratio}")


async def filter_hostnames() -> None:
    """
    from clickhouse vps dns mapping table, extract geo information from hostname resolution
    valid hostname: hostname that carry geo information
    invalid hostname: hostname that does not carry geo information
    """
    (
        valid_hostnames,
        invalid_hostnames,
    ) = await get_hostnames_geographic_influence()

    dump_csv(
        [hostname for hostname in valid_hostnames["country"]],
        path_settings.DATASET / "valid_hostnames.csv",
    )
    dump_csv(
        [hostname for hostname in invalid_hostnames["country"]],
        path_settings.DATASET / "invalid_hostnames.csv",
    )

    # get rows with valid hostname
    # valid_hostnames_rows = Get().get_valid_hostnames(
    #     table_name=clickhouse_settings.DNS_MAPPING_VPS_RAW,
    #     valid_hostnames=[hostname for hostname, _, _ in valid_hostnames["country"]],
    # )

    # create_table_statement = DNSMappingTable().create_table_statement(
    #     table_name="filtered_hostname_mapping"
    # )

    # DNSMappingTable().insert(
    #     input_data=valid_hostnames_rows,
    #     table_name="filtered_hostname_mapping",
    #     create_table_statement=create_table_statement,
    # )

    print_hostname_info(valid_hostnames, invalid_hostnames)


async def filter_ecs_hostnames() -> None:
    """perform zdns resolution on host subnet, remove hostnames with source scope equal to 0"""
    host_addr = get_host_ip_addr()
    host_subnet = get_prefix_from_ip(host_addr)

    await resolve_hostnames(
        subnets=[host_subnet],
        hostname_file=path_settings.HOSTNAMES_MILLIONS,
        output_table=clickhouse_settings.DNS_MAPPING_ECS,
        repeat=False,
        end_date=None,
    )


async def filter_anycast_hostnames() -> None:
    """get all answers, compare with anycatch database"""
    anycatch_db = load_anycatch_data()
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        ecs_hostnames = await GetDNSMapping().execute(
            client=client, table_name=clickhouse_settings.DNS_MAPPING_ECS
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


async def resolve_vps_subnet() -> None:
    """perform ECS-DNS resolution one all VPs subnet"""
    unicast_hostnames = await filter_anycast_hostnames()
    unicast_hostnames = list(unicast_hostnames)

    tmp_hostname_file = create_tmp_csv_file(unicast_hostnames)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps_subnet = await GetSubnets().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )

    await resolve_hostnames(
        subnets=[row["subnet"] for row in vps_subnet],
        hostname_file=tmp_hostname_file,
        output_table=clickhouse_settings.DNS_MAPPING_VPS_RAW,
        repeat=False,
        end_date=None,
        chunk_size=1_000,
    )

    tmp_hostname_file.unlink()


async def main() -> None:
    """init main"""
    # logger.info("Starting Geolocation hostnames initialization")

    # await filter_ecs_hostnames()

    # logger.info("Retrieved ECS hostnames")

    # await filter_anycast_hostnames()

    logger.info("Filtered anycast hostnames")

    # await resolve_vps_subnet()

    logger.info("Hostname resolution done for every VPs")

    await filter_hostnames()

    logger.info("Geographically valid hostname done")

    # await filter_greedy_hostname()

    # await filter_cdn_diversity_hostname()


if __name__ == "__main__":
    asyncio.run(main())
