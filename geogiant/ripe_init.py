"""
create database tables, 
retrieve vantage points, 
retrieve RIPE Atlas public traceroutes,
evaluate probe connectivity,
insert all data 
"""

import httpx
import json
import asyncio

from pyasn import pyasn
from ipaddress import IPv4Address
from typing import Generator
from pathlib import Path
from numpy import mean
from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient


from geogiant.prober import RIPEAtlasAPI
from geogiant.clickhouse import (
    GetVPs,
    GetSubnets,
    CreateGeolocTable,
    CreateTracerouteTable,
    InsertFromCSV,
    GetProbeConnectivity,
    GetGeolocFromTraceroute,
)

from geogiant.common.geoloc import distance, rtt_to_km
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.files_utils import load_countries_info, create_tmp_csv_file
from common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def init_ripe_atlas_prober(output_table: str) -> None:
    """insert vps within clickhouse db"""
    api = RIPEAtlasAPI()
    vps = await api.get_vps()
    await api.insert_vps(vps, output_table)


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


def filter_vps(vps: list) -> None:
    """filter vps based on 1) default geolocation 2) DNS resolution"""
    # 1. filter default location VPs
    vps = filter_default_geoloc(vps)

    # 4. TODO: get resolution for all VPs on a set of hostnames
    return vps


def parse_traceroute(traceroute: dict) -> str:
    """retrieve all measurement, parse data and return for clickhouse insert"""
    traceroute_results = []
    for ttl_results in traceroute["result"]:
        ttl = ttl_results["hop"]

        # hop 255, junk data
        if ttl == 255:
            continue

        # retrieve rtt from response
        rcvd = 0
        sent = len(ttl_results["result"])
        responses = defaultdict(list)
        for resp in ttl_results["result"]:
            if "rtt" in resp:
                responses[resp["from"]].append(resp["rtt"])
                rcvd += 1

        # no response for current TTL
        if not responses:
            continue

        for ip_addr, rtts in responses.items():
            # remove private ip addresses
            if IPv4Address(ip_addr).is_private:
                continue

            min_rtt = min(rtts)
            max_rtt = max(rtts)
            avg_rtt = mean(rtts)

            traceroute_results.append(
                f"{traceroute['timestamp']},\
                    {traceroute['from']},\
                    {get_prefix_from_ip(traceroute['from'])},\
                    {24},\
                    {traceroute['prb_id']},\
                    {traceroute['msm_id']},\
                    {traceroute['dst_addr']},\
                    {get_prefix_from_ip(traceroute['dst_addr'])},\
                    {traceroute['proto']},\
                    {ip_addr},\
                    {get_prefix_from_ip(ip_addr)},\
                    {ttl},\
                    {rcvd},\
                    {sent},\
                    {min_rtt},\
                    {max_rtt},\
                    {avg_rtt},\
                    \"{rtts}\""
            )

    return traceroute_results


def load_iter(in_file: Path) -> Generator:
    """load iter large file"""
    for row in in_file.open("r"):
        yield json.loads(row)


async def download() -> None:
    """download public measurement from RIPE Atlas FTP, output results into clickhouse"""
    base_url = "https://data-store.ripe.net/datasets/atlas-daily-dumps/"
    dump_url = "2024-02-06/"
    file_url = base_url + dump_url + "traceroute-2024-02-06T2300.bz2"
    output_file = path_settings.DATASET / "traceroute-2024-02-06T2300.bz2"

    async with httpx.AsyncClient() as client:
        resp = await client.get(file_url)
        print(file_url)

    with output_file.open("wb") as f:
        f.write(resp.content)


async def insert() -> None:
    """from a decompressed traceroute file, parse them"""
    in_file = path_settings.DATASET / "traceroute-2024-02-06T2300"
    batch_size = 100_000
    traceroutes = []

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreateTracerouteTable().execute(
            client, clickhouse_settings.RIPE_ATLAS_TRACEROUTES
        )

    for traceroute in load_iter(in_file):

        # remove IPv6 and corrupted traceroutes
        if (
            traceroute["af"] != 4
            or not traceroute["from"]
            or not traceroute["destination_ip_responded"]
        ):
            continue

        traceroute = parse_traceroute(traceroute)
        traceroutes.extend(traceroute)

        if len(traceroutes) > batch_size:
            logger.info(f"inserting batch of traceroute: limit = {batch_size}")

            tmp_file_path = create_tmp_csv_file(traceroutes)

            await InsertFromCSV().execute(
                table_name=clickhouse_settings.RIPE_ATLAS_TRACEROUTES,
                in_file=tmp_file_path,
            )

            tmp_file_path.unlink()
            traceroutes = []


async def probe_connectivity() -> None:
    """get VPs connectivity based on avg first reply rtt from RIPE Atlas public traceroutes"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        i = 0
        resp = await GetProbeConnectivity().execute_iter(
            client=client,
            table_name=clickhouse_settings.RIPE_ATLAS_TRACEROUTES,
        )

        async for row in resp:
            vp = row["src_addr"]
            connectivity = row["connectivity"]
            logger.info(f"vp = {vp}, connectivity = {connectivity}")

            i += 1
            if i > 10:
                break

    # TODO: get entire VPs table, add connectivity


async def get_geoloc_from_traceroute() -> None:
    """
    retrieve all IPs with latency below a given threshold
    (default = 2ms) from RIPE Atlas public traceroutes
    """
    asndb = pyasn(str(path_settings.RIB_TABLE))

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = {}
        resp = await GetVPs().execute_iter(
            client=client,
            table_name=clickhouse_settings.VPS_RAW,
        )
        async for row in resp:
            vps[row["vp_addr"]] = row

        geoloc_ip_addrs = []
        resp = await GetGeolocFromTraceroute().execute_iter(
            client=client,
            table_name=clickhouse_settings.RIPE_ATLAS_TRACEROUTES,
        )

        async for row in resp:

            addr = row["reply_addr"]
            subnet = get_prefix_from_ip(addr)
            asn, bgp_prefix = route_view_bgp_prefix(addr, asndb)

            # filter anycast IP addrs

            # TODO: Get all VPs present in measurement
            vp_addr = row["src_addr"]
            try:
                vp_info = vps[vp_addr]
            except KeyError:
                logger.info(f"VP::{vp_addr} not in VPs table")
                continue

            # TODO: get asn and bgp prefix from RIPE if not found?
            if not asn:
                asn = -1
            if not bgp_prefix:
                bgp_prefix = "Unknown"

            min_rtt = row["min_rtt"]
            measured_dst = rtt_to_km(min_rtt)

            geoloc_ip_addrs.append(
                f"{row['reply_addr']},\
                {subnet},\
                {bgp_prefix},\
                {asn},\
                {vp_info['lat']},\
                {vp_info['lon']},\
                {vp_info['country_code']},\
                {vp_addr},\
                {vp_info['vp_subnet']},\
                {vp_info['vp_bgp_prefix']},\
                {min_rtt},\
                {measured_dst}"
            )

        await CreateGeolocTable().execute(
            client=client,
            table_name=clickhouse_settings.RIPE_ATLAS_TRACEROUTE_GEOLOC,
        )

    tmp_file = create_tmp_csv_file(geoloc_ip_addrs)
    await InsertFromCSV().execute(
        table_name=clickhouse_settings.RIPE_ATLAS_TRACEROUTE_GEOLOC,
        in_file=tmp_file,
    )

    tmp_file.unlink()


async def main() -> None:
    """init main"""
    logger.info("Starting RIPE Atlas prober initialization")
    vps_table = clickhouse_settings.VPS_RAW

    await init_ripe_atlas_prober(output_table=vps_table)

    # await download()

    # await insert()

    # await probe_connectivity()
    # await get_geoloc_from_traceroute()


if __name__ == "__main__":
    asyncio.run(main())
