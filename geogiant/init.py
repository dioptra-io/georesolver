"""create database tables, retrieve vantage points, perform zdns / zmap measurement and populate database"""
import asyncio

from loguru import logger
from pyasn import pyasn

from clickhouse import VPsTable, Get
from prober import RIPEAtlasAPI
from zdns.zdns_wrapper import ZDNS

from common.geoloc import distance
from common.files_utils import load_countries_default_geoloc, load_countries_continent
from common.ip_addresses_utils import route_view_bgp_prefix, get_prefix_from_ip
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def filter_default_geoloc(vps: list, min_dist_to_default: int = 10) -> dict:
    """filter vps with coordinates too close from their country's default location"""
    countries = load_countries_default_geoloc()

    valid_vps = []
    for vp in vps:
        try:
            country_geo = countries[vp["country_code"]]
        except KeyError:
            logger.warning(f"error country code {vp['country_code']} is unknown")
            valid_vps.append(vp)
            continue

        dist = distance(
            country_geo["lat"],
            vp["lat"],
            country_geo["lon"],
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


def insert_vps(vps: list[dict], table_name: str) -> None:
    """insert vps within clickhouse db"""
    asndb = pyasn(str(path_settings.RIB_TABLE))

    csv_data = []
    for vp in vps:
        _, bgp_prefix = route_view_bgp_prefix(vp["address_v4"], asndb)
        subnet = get_prefix_from_ip(vp["address_v4"])

        csv_data.append(
            f"{vp['address_v4']},\
            {subnet},\
            {vp['asn_v4']},\
            {bgp_prefix},\
            {vp['country_code']},\
            {vp['lat']},\
            {vp['lon']},\
            {vp['id']},\
            {vp['is_anchor']}"
        )

    create_table_statement = VPsTable().create_table_statement(
        table_name=table_name,
    )

    VPsTable().insert(
        input_data=csv_data,
        table_name=table_name,
        create_table_statement=create_table_statement,
        drop_table=True,
    )


async def init_prober(api: RIPEAtlasAPI) -> None:
    """get connected vps from measurement platform, insert in clickhouse"""
    vps = await api.get_vps()
    insert_vps(vps, clickhouse_settings.VPS_RAW)


# TODO: async DNS
def get_raw_dns_mapping() -> None:
    """perform DNS mapping with zdns on VPs subnet"""
    vps_subnet = Get().subnet(clickhouse_settings.VPS_RAW)
    vps_subnet = [subnet + "/24" for subnet in vps_subnet]

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
    subnet_per_hostname = Get().subnet_per_hostname(
        clickhouse_settings.DNS_MAPPING_VPS_RAW
    )
    vps_per_subnet = Get().vps_country_per_subnet(clickhouse_settings.VPS_RAW)
    countries_continent = load_countries_continent()

    hostnames = []
    invalid_hostnames = []
    for hostname in subnet_per_hostname.items():
        for answer_bgp_prefix, subnets in subnet_per_hostname[hostname].items():
            # get vps continent
            mapping_continents = set()
            for subnet in subnets:
                for vp in vps_per_subnet[subnet]:
                    continent = countries_continent[vp["country_code"]]

                    mapping_continents.add(continent)

            if len(mapping_continents) > 1:
                logger.error(
                    f"{hostname}:{answer_bgp_prefix}::DNS mapping on multiple continent"
                )
                invalid_hostnames.append(hostname)
            else:
                hostnames.append(hostname)

    # load all VPs raw DNS mapping data, filter, insert results

    logger.info(
        f"Invalid hostnames = {len(invalid_hostnames)} | remaining = {len(hostnames)}"
    )


async def main() -> None:
    """init main"""
    api = RIPEAtlasAPI()

    # await init_prober(api=api)

    get_raw_dns_mapping()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
