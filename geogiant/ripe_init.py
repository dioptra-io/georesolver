"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""
import asyncio

from loguru import logger
from pych_client import ClickHouseClient

from geogiant.clickhouse import Get, GetSubnets, CreateVPsTable
from geogiant.prober import RIPEAtlasAPI
from geogiant.zdns import ZDNS

from geogiant.common.files_utils import create_tmp_csv_file
from geogiant.common.ip_addresses_utils import route_view_bgp_prefix, get_prefix_from_ip
from geogiant.common.geoloc import distance
from geogiant.common.files_utils import (
    load_countries_info,
    load_anycatch_data,
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


# TODO: async DNS
def get_raw_dns_mapping() -> None:
    """perform DNS mapping with zdns on VPs subnet"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps_subnet = GetSubnets().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )

    vps_subnet = [subnet["subnet"] + "/24" for subnet in vps_subnet]

    zdns = ZDNS(
        subnets=vps_subnet,
        hostname_file=path_settings.CDN_HOSTNAMES_RAW,
        name_servers="8.8.8.8",
        table_name=clickhouse_settings.DNS_MAPPING_VPS_RAW,
    )
    zdns.main()


def filter_vps(vps: list) -> None:
    """filter vps based on 1) default geolocation 2) DNS resolution"""
    # 1. filter default location VPs
    vps = filter_default_geoloc(vps)

    # 4. TODO: get resolution for all VPs on a set of hostnames
    return vps


def filter_hostnames() -> None:
    """filter hostnames based on VPs DNS mapping"""
    anycast_prefixes = load_anycatch_data()

    subnet_per_hostname = Get().subnet_per_hostname(
        clickhouse_settings.DNS_MAPPING_VPS_RAW
    )
    vps_per_subnet = Get().vps_country_per_subnet(clickhouse_settings.VPS_RAW)
    countries = load_countries_info()

    hostnames = set()
    invalid_hostnames = set()
    for (hostname, answer_bgp_prefix), subnets in subnet_per_hostname.items():
        # get vps continent

        if set(answer_bgp_prefix).intersection(anycast_prefixes):
            logger.info(
                f"{hostname}:{answer_bgp_prefix}::Detected as anycast by anycatch db"
            )
            invalid_hostnames.add(hostname)

        hostnames.add(hostname)

        mapping_continents = set()
        vp_countries = set()
        for subnet in subnets:
            for vp in vps_per_subnet[subnet]:
                country = countries[vp["country_code"]]
                vp_countries.add(vp["country_code"])
                continent = country["continent"]
                mapping_continents.add(continent)

        if len(mapping_continents) >= 4:
            logger.error(
                f"{hostname}:{answer_bgp_prefix}::DNS mapping on multiple continent, {mapping_continents}"
            )
            logger.error(f"vps country:{[c for c in vp_countries]}")
            invalid_hostnames.add(hostname)

    hostnames = hostnames.difference(invalid_hostnames)

    # load all VPs raw DNS mapping data, filter, insert results
    logger.info(f"Valid hostnames = {len(hostnames)}")
    for hostname in hostnames:
        logger.info(f"Hostname: {hostname}")

    logger.info(f"Invalid Valid hostnames = {len(invalid_hostnames)}")
    for hostname in invalid_hostnames:
        logger.info(f"Hostname: {hostname}")


async def main() -> None:
    """init main"""
    logger.info("Starting RIPE Atlas prober initialization")

    await init_ripe_atlas_prober()

    # get_raw_dns_mapping()

    # filter_hostnames()


if __name__ == "__main__":
    asyncio.run(main())
