"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""

import json
import asyncio
import schedule
import time

from pyasn import pyasn
from tqdm import tqdm
from pathlib import Path
from numpy import mean
from loguru import logger
from pych_client import AsyncClickHouseClient
from ipwhois import IPWhois
from collections import defaultdict

from geogiant.zdns import ZDNS
from geogiant.clickhouse import (
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
    dump_csv,
    create_tmp_csv_file,
    load_csv,
    dump_json,
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
    chunk_size: int = 100,
) -> None:
    """repeat zdns measurement on set of VPs"""

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
                    table_name=output_table,
                    name_servers="8.8.8.8",
                    iterative=True,
                    timeout=0,
                )
                await zdns.main()

                tmp_hostname_file.unlink()

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


async def filter_geo_hostnames() -> tuple[dict, dict]:
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

    valid_hostnames = load_csv(path_settings.DATASET / "valid_hostnames.csv")
    invalid_hostnames = load_csv(path_settings.DATASET / "invalid_hostnames.csv")
    treated_hostnames = valid_hostnames + invalid_hostnames
    valid_hostnames = set(valid_hostnames)
    invalid_hostnames = set(invalid_hostnames)
    treated_hostnames = set([row.split(",")[0] for row in treated_hostnames])

    logger.info(f"{len(treated_hostnames)}:: Hostnames were already analyzed")

    for row in tqdm(subnet_per_hostname):
        hostname = row["hostname"]
        vps_per_bgp_prefix = row["vps_per_bgp_prefix"]

        if hostname in treated_hostnames:
            logger.debug(f"{hostname} already analyzed")
            continue

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
        chunk_size=10_000,
    )


async def filter_anycast_hostnames(input_table: str) -> None:
    """get all answers, compare with anycatch database"""
    anycatch_db = load_anycatch_data()
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        ecs_hostnames = await GetDNSMapping().execute(
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


async def resolve_vps_subnet(input_table: str, output_table: str) -> None:
    """perform ECS-DNS resolution one all VPs subnet"""
    unicast_hostnames = await filter_anycast_hostnames(input_table)
    unicast_hostnames = list(unicast_hostnames)

    tmp_hostname_file = create_tmp_csv_file(unicast_hostnames)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps_subnet = await GetSubnets().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )

    await resolve_hostnames(
        subnets=[row["subnet"] for row in vps_subnet],
        hostname_file=tmp_hostname_file,
        output_table=output_table,
        repeat=False,
        end_date=None,
        chunk_size=10_000,
    )

    tmp_hostname_file.unlink()


async def get_hostname_cdn() -> None:
    """for each IP address returned by a hostname, retrieve the CDN behind"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    # hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames.csv")
    # hostname_filter = [row.split(",")[0] for row in hostname_filter]

    asn_to_org = {}
    with (path_settings.DATASET / "20240101.as-org2info.jsonl").open("r") as f:
        for row in f.readlines():
            row = json.loads(row)
            if "asn" in row and "name" in row:
                asn_to_org[int(row["asn"])] = row["name"]

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetHostnamesAnswerSubnet().execute_iter(
            client=client,
            table_name="filtered_hostnames_ecs_mapping",
            hostname_filter="",
        )

        org_per_hostname = defaultdict(dict)
        async for row in resp:
            hostname = row["hostname"]
            answers = row["answer"]

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

            # logger.info(f"{hostname=}, {org_per_hostname[hostname]=}")

        # set are not json compatible
        for hostname in org_per_hostname:
            for org, bgp_prefixes in org_per_hostname[hostname].items():
                org_per_hostname[hostname][org] = list(set(bgp_prefixes))

        dump_json(
            org_per_hostname, path_settings.DATASET / "hostname_1M_organization.json"
        )


async def main() -> None:
    """init main"""
    logger.info("Starting Geolocation hostnames initialization")

    # await filter_ecs_hostnames(output_table="zdns_resolver_ecs_CrUX")

    # logger.info("Retrieved ECS hostnames")

    # await filter_anycast_hostnames()

    logger.info("Filtered anycast hostnames")

    await resolve_vps_subnet(
        input_table="zdns_resolver_ecs_CrUX", output_table="filtered_hostnames_ecs_zdns"
    )

    logger.info("Hostname resolution done for every VPs")

    # await filter_geo_hostnames()

    # logger.info("Geographically valid hostname done")

    # await get_hostname_cdn()


if __name__ == "__main__":
    asyncio.run(main())
