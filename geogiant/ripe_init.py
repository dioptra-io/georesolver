"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""
import asyncio
import schedule
import time

from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass
from numpy import mean
from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import (
    GetSubnets,
    GetSubnetPerHostname,
    GetVPSInfoPerSubnet,
    GetHostnames,
    GetDNSMapping,
)
from geogiant.prober import RIPEAtlasAPI
from geogiant.zdns import ZDNS

from geogiant.common.geoloc import distance
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


async def init_ripe_atlas_prober() -> None:
    """insert vps within clickhouse db"""
    api = RIPEAtlasAPI()
    vps = await api.get_vps()
    await api.insert_vps(vps, clickhouse_settings.VPS_RAW)


def filter_default_geoloc(vps: list, min_dist_to_default: int = 10) -> dict:
    """filter vps with coordinates too close from their country's default location"""
    countries = load_countries_info()

    valid_vps = []
    for vp in vps:
        try:
            country_geo = countries[vp["country_code"]]
        except KeyError:
            logger.warning(f"error country code {vp['country_code']} is unknown")
            continue

        dist = distance(
            country_geo["default_lat"],
            vp["lat"],
            country_geo["default_lon"],
            vp["lon"],
        )

        # Keep VPs that are away from default country geolocation
        if dist > min_dist_to_default:
            valid_vps.append(vp)
        else:
            logger.info(
                f"{vp['address_v4']}/{vp['id']}::Probed removed because of default geolocation"
            )

    return valid_vps


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

    # TODO: harmonize logs
    # logger.remove()
    # logger.add(path_settings.LOG_PATH / "resolve_hostnames.log")

    async def raw_dns_mapping(subnets: list) -> None:
        """perform DNS mapping with zdns on VPs subnet"""

        subnets = [subnet + "/24" for subnet in subnets]

        with hostname_file.open("r") as f:
            logger.info("raw hostname file already generated")

            # take max hostnames from hostname file
            all_hostnames = f.readlines()
            all_hostnames = all_hostnames[:max_hostname]

            # split file
            partial_hostname_files = []
            files_uuid = uuid4()
            for index in range(0, len(all_hostnames), chunk_size):
                file_path = (
                    path_settings.TMP_PATH
                    / f"hostnames_part_{index // chunk_size}_{files_uuid}.csv"
                )

                hostnames = all_hostnames[index : index + chunk_size]

                with file_path.open("w") as f:
                    for hostname in hostnames:
                        f.write(hostname)

                # save file path
                partial_hostname_files.append(file_path)

        for i, file_path in enumerate(partial_hostname_files):
            logger.info(
                f"Starting to resolve hostnames {i * chunk_size} to {(i + 1) * chunk_size} (total: {len(partial_hostname_files * chunk_size)})"
            )

            zdns = ZDNS(
                subnets=subnets,
                hostname_file=file_path,
                name_servers="8.8.8.8",
                table_name=output_table,
            )
            await zdns.main()

            # remove tmp file
            file_path.unlink()
            time.sleep(1)

    await raw_dns_mapping(subnets)

    if repeat:
        # TODO: replace with Crontab module
        schedule.every(4).hours.until(end_date).do(raw_dns_mapping)

        while True:
            schedule.run_pending()


def filter_vps(vps: list) -> None:
    """filter vps based on 1) default geolocation 2) DNS resolution"""
    # 1. filter default location VPs
    vps = filter_default_geoloc(vps)

    # 4. TODO: get resolution for all VPs on a set of hostnames
    return vps


def in_anycatch(bgp_prefix: str, anycast_prefixes: list) -> bool:
    """check if a bgp prefix is anycast or not based on anycatch data"""
    if set(bgp_prefix).intersection(anycast_prefixes):
        return True
    return False


def get_geographic_mapping(
    subnets: dict, vps_per_subnet: dict, countries_info: dict
) -> tuple[list, list]:
    """get vps country and continent for a given hostname bgp prefix"""
    mapping_continents = []
    mapping_countries = []
    for subnet in subnets:
        for vp in vps_per_subnet[subnet]:
            country_code = vp[-1]
            continent = countries_info[country_code]["continent"]

            mapping_countries.append(country_code)
            mapping_continents.append(continent)

    return mapping_countries, mapping_continents


def get_geographic_ratio(geographic_mapping: list) -> float:
    """return the ratio of the most represented geographic granularity (country/continent)"""
    return max(
        [
            (region, geographic_mapping.count(region) / len(geographic_mapping))
            for region in geographic_mapping
        ],
        key=lambda x: x[-1],
    )


@dataclass
class HostnameMappingGeo:
    bgp_prefix: str
    countries: set()
    continents: set()
    major_country: str
    major_country_ratio: float
    major_continent: str
    major_continent_ratio: float


def filter_on_geo_distribution(
    hostnames_geo_mapping: dict[list],
    threshold_country: float = 0.7,
    threshold_continent: float = 0.9,
) -> dict:
    """
    get hostnames and their major geographical region of influence.
    Return True if the average ratio for each hostname's BGP prefix is above
    a given threshold (80% of VPs in the same region by default)
    """
    valid_hostnames = defaultdict(list)
    invalid_hostnames = defaultdict(list)
    hostname_cov_countries = set()
    hostname_cov_continents = set()
    for hostname, geo_mappings in hostnames_geo_mapping.items():
        avg_ratio_country = []
        avg_ratio_continent = []
        mapping: HostnameMappingGeo = None
        for mapping in geo_mappings:
            avg_ratio_country.append(mapping.major_country_ratio)
            avg_ratio_continent.append(mapping.major_continent_ratio)

            for country in mapping.countries:
                hostname_cov_countries.add(country)

            for continent in mapping.continents:
                hostname_cov_continents.add(continent)

        avg_ratio_country = mean(avg_ratio_country)
        if avg_ratio_country > threshold_country:
            valid_hostnames["country"].append(
                (hostname, hostname_cov_countries, avg_ratio_country)
            )
        else:
            invalid_hostnames["country"].append(
                (hostname, hostname_cov_countries, avg_ratio_country)
            )

        avg_ratio_continent = mean(avg_ratio_continent)
        if avg_ratio_continent > threshold_continent:
            valid_hostnames["continent"].append(
                (hostname, hostname_cov_continents, avg_ratio_continent)
            )
        else:
            invalid_hostnames["continent"].append(
                (hostname, hostname_cov_continents, avg_ratio_continent)
            )

    return valid_hostnames, invalid_hostnames


async def get_hostnames_geographic_influence() -> tuple[dict, dict]:
    """
    for each hostname, get vps mapped per BGP prefix and get hostname geographical area of influence
    Assumption: hostnames that perform ECS-DNS resolution based on geographical information
    associate BGP prefixes to vps in the same region (either country or continent).
    """
    anycatch_prefixes = load_anycatch_data()
    countries_info = load_countries_info()

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetVPSInfoPerSubnet().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )
        vps_per_subnet = {}
        for row in resp:
            vps_per_subnet[row["subnet"]] = row["vps"]

        subnet_per_hostname = await GetSubnetPerHostname().execute(
            client=client, table_name=clickhouse_settings.DNS_MAPPING_VPS_RAW
        )

    anycatch_filter = set()
    hostname_mapping_geo = defaultdict(list)

    for row in subnet_per_hostname:
        hostname = row["hostname"]
        answer_bgp_prefix = row["answer_bgp_prefix"]
        subnets = row["subnets"]

        if in_anycatch(answer_bgp_prefix, anycatch_prefixes):
            anycatch_filter.add(hostname)
            continue

        # get the set of countries and continent present in the mapping
        # TODO: compute distance?
        mapping_countries, mapping_continents = get_geographic_mapping(
            subnets, vps_per_subnet, countries_info
        )

        major_country, major_country_ratio = get_geographic_ratio(mapping_countries)
        major_continent, major_continent_ratio = get_geographic_ratio(
            mapping_continents
        )

        # get and store hostname bgp prefix analysis
        hostname_mapping_geo[hostname].append(
            HostnameMappingGeo(
                bgp_prefix=answer_bgp_prefix,
                countries=set(mapping_countries),
                continents=set(mapping_continents),
                major_country=major_country,
                major_country_ratio=major_country_ratio,
                major_continent=major_continent,
                major_continent_ratio=major_continent_ratio,
            )
        )

    # filter hostnames based on geographic distribution
    valid_hostnames, invalid_hostnames = filter_on_geo_distribution(
        hostname_mapping_geo,
        threshold_country=0.7,
        threshold_continent=0.9,
    )

    return valid_hostnames, invalid_hostnames


def print_hostname_info(
    valid_hostnames: dict[list], invalid_hostnames: dict[list]
) -> None:
    """print info on hostname filtering analysis"""
    # load all VPs raw DNS mapping data, filter, insert results
    logger.info("##########################################################")
    logger.info("# HOSTNAMES GEO VALIDATION: COUNTRY EVAL #################")
    logger.info("##########################################################")
    logger.info(f"Valid hostnames = {len(valid_hostnames['country'])}")
    for hostname, cov, avg_ratio in valid_hostnames["country"]:
        logger.info(
            f"Hostname = {hostname} coverage =  {len(cov)} avg ratio = {avg_ratio}"
        )

    logger.info("\n")

    logger.info(f"Invalid hostnames = {len(invalid_hostnames['country'])}")
    for hostname, country_cov, avg_ratio in invalid_hostnames["country"]:
        logger.info(
            f"Hostname = {hostname} coverage =  {len(cov)} avg ratio = {avg_ratio}"
        )

    logger.info("##########################################################")
    logger.info("# HOSTNAMES GEO VALIDATION: CONTINENT EVAL ###############")
    logger.info("##########################################################")
    for hostname, cov, avg_ratio in valid_hostnames["continent"]:
        logger.info(
            f"Hostname = {hostname}  coverage =  {len(cov)} avg ratio = {avg_ratio}"
        )

    logger.info("\n")

    logger.info(f"Invalid hostnames = {len(invalid_hostnames['continent'])}")
    for hostname, country_cov, avg_ratio in invalid_hostnames["continent"]:
        logger.info(
            f"Hostname = {hostname} coverage =  {len(country_cov)} avg ratio = {avg_ratio}"
        )


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
        [hostname for hostname, _, _ in valid_hostnames["country"]],
        path_settings.DATASET / "valid_hostnames.csv",
    )
    dump_csv(
        [hostname for hostname, _, _ in invalid_hostnames["country"]],
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
        answer = row["answer"]
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
    )


async def main() -> None:
    """init main"""
    logger.info("Starting RIPE Atlas prober initialization")

    # await init_ripe_atlas_prober()

    # await filter_ecs_hostnames()

    # await filter_anycast_hostnames()

    await resolve_vps_subnet()

    # await filter_hostnames()


if __name__ == "__main__":
    asyncio.run(main())
