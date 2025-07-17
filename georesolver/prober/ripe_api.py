import time
import json
import httpx
import pyasn
import asyncio

from numpy import mean
from collections import defaultdict
from ipaddress import IPv4Address, IPv6Address, AddressValueError

from enum import Enum
from loguru import logger
from pych_client import AsyncClickHouseClient
from pych_client.exceptions import ClickHouseException

from georesolver.clickhouse import (
    DropTable,
    InsertCSV,
    CreateVPsTable,
    CreatePingTable,
    CreateTracerouteTable,
)
from georesolver.clickhouse.queries import load_vps
from georesolver.common.files_utils import create_tmp_csv_file, dump_pickle
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.settings import RIPEAtlasSettings


class RIPEAtlasStatus(Enum):
    SCHEDULED: str = "SCHEDULED"
    ONGOING: str = "ONGOING"
    STOPPED: str = "STOPPED"


class RIPEAtlasAPI:
    """RIPE Atlas measurement API"""

    def __init__(self) -> None:
        self.settings = RIPEAtlasSettings()
        self.api_url = "https://atlas.ripe.net/api/v2"
        self.measurement_url = f"{self.api_url}/measurements/"
        self.headers = {
            "Authorization": f"Key {RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY}"
        }
        self.request_timeout = 60

    def check_schedule_validity(self, schedule: list[tuple]) -> None:
        """for any target, check if not too many VPs are scheduled"""
        for target, vp_ids in schedule:
            if len(vp_ids) > self.settings.MAX_VP:
                raise RuntimeError(
                    f"Too many VPs scheduled for target: {target} (nb vps: {len(vp_ids)}, max: {self.settings.MAX_VP})"
                )

    def get_ping_config(
        self,
        target: str,
        vp_ids: list[int],
        probing_tag: str,
        protocol: str = "ICMP",
        ipv6: bool = False,
    ) -> dict:
        return {
            "definitions": [
                {
                    "target": target,
                    "af": 4 if not ipv6 else 6,
                    "packets": self.settings.PING_NB_PACKETS,
                    "proto": protocol,
                    "size": 48,
                    "tags": ["dioptra", str(probing_tag)],
                    "description": f"Active Geolocation of {target}",
                    "resolve_on_probe": False,
                    "skip_dns_check": True,
                    "include_probe_id": False,
                    "type": "ping",
                }
            ],
            "probes": [
                {"value": v_id, "type": "probes", "requested": 1} for v_id in vp_ids
            ],
            "is_oneoff": True,
        }

    def get_traceroute_config(
        self,
        target: str,
        vp_ids: list[int],
        probing_tag: str,
        min_ttl: int = 1,
        max_hops: int = 32,
        protocol: str = "ICMP",
        ipv6: bool = False,
    ) -> dict:
        return {
            "definitions": [
                {
                    "target": target,
                    "af": 4 if not ipv6 else 6,
                    "packets": self.settings.PING_NB_PACKETS,
                    "first_hop": min_ttl,
                    "max_hops": max_hops,
                    "protocol": protocol,
                    "tags": ["dioptra", str(probing_tag)],
                    "description": f"Dioptra Traceroute of {target}",
                    "resolve_on_probe": False,
                    "skip_dns_check": True,
                    "include_probe_id": False,
                    "type": "traceroute",
                }
            ],
            "probes": [
                {"value": v_id, "type": "probes", "requested": 1} for v_id in vp_ids
            ],
            "is_oneoff": True,
        }

    def is_geoloc_disputed(self, probe: dict) -> bool:
        """check if geoloc disputed flag is contained in probe metadata"""
        tags = probe["tags"]
        for tag in tags:
            if tag["slug"] == "system-geoloc-disputed":
                return True
        return False

    async def get_ongoing_measurements(self, tag: str, wait_time: int = 30) -> int:
        """return the number of measurements which have the status Ongoing"""
        async with httpx.AsyncClient() as client:
            params = {
                "status__in": "Specified,Scheduled,Ongoing",
                "tags": [tag],
                "mine": True,
            }
            try:
                resp = await client.get(
                    self.measurement_url,
                    params=params,
                    headers=self.headers,
                    timeout=self.request_timeout,
                )
                resp = resp.json()
            except httpx.ReadTimeout:
                await asyncio.sleep(wait_time)
                return None
            except Exception as e:
                logger.error(f"Unsuported error:: {e}")
                await asyncio.sleep(wait_time)
                return None
        try:
            ongoing_measurements = resp["count"]
            logger.info(f"Nb ongoing measurements:: {ongoing_measurements}")
        except KeyError:
            raise RuntimeError(f"Count key should be present in response:: {resp}")

        return ongoing_measurements

    async def stop_measurement(self, id: int) -> None:
        """stop an ongoing measurement"""
        async with httpx.AsyncClient() as client:
            params = {"id": id}
            try:
                resp = await client.delete(
                    url=self.measurement_url + f"{id}",
                    params=params,
                    headers=self.headers,
                )
            except httpx.ReadTimeout:
                return id

            # delete response is empty
            if resp.status_code != 204:
                logger.error(f"Delete query failed because:: {resp.json()}")

            return id

    async def get_stopped_measurement_ids(
        self,
        start_time: str,
        tags: list[str],
        wait_time: int = 30,
    ) -> list[int]:
        """return all measurements with a status stopped"""

        params = {
            "sort": ["start_time"],
            "status__in": "Stopped,Forced to stop,No suitable probes,Failed",
            "stop_time__gte": start_time,
            "tags": [str(tag) for tag in tags],
            "mine": True,
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    self.measurement_url, params=params, headers=self.headers
                )
                if resp.status_code != 200:
                    logger.error(f"Error:: {resp.json()}")

                resp = resp.json()
            except httpx.ReadTimeout:
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Unsuported error:: {e}")
                await asyncio.sleep(wait_time)

        stopped_measurements = [m["id"] for m in resp["results"]]

        while resp["next"]:
            async with httpx.AsyncClient() as client:
                for _ in range(3):
                    try:
                        resp = await client.get(resp["next"], params=params)
                        break
                    except httpx.ReadTimeout:
                        await asyncio.sleep(wait_time)
                    except Exception as e:
                        logger.error(f"Unsuported error:: {e}")
                        await asyncio.sleep(wait_time)

                resp = resp.json()

            stopped_measurements.extend([m["id"] for m in resp["results"]])

        return stopped_measurements

    async def get_raw_vps(self, url: str = "https://atlas.ripe.net/api/v2/probes/"):
        """get request url atlas endpoint"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={"status": [1]})
            resp = resp.json()

            for vp in resp["results"]:
                yield vp

            await asyncio.sleep(0.1)

            while url := resp.get("next"):
                logger.debug(f"Next page::{resp['next']}")
                resp = await client.get(resp["next"])
                resp = resp.json()

                for vp in resp["results"]:
                    yield vp

                await asyncio.sleep(0.1)

    async def get_vps(self, probes_only: bool = False, ipV6: bool = False) -> list:
        """return all RIPE Atlas VPs (set probe_only to remove anchors)"""
        vps = []
        rejected = 0
        vp: dict = None
        async for vp in self.get_raw_vps():
            if vp["is_anchor"] and probes_only:
                continue
            else:
                if (
                    vp["status"]["name"] != "Connected"
                    or vp.get("geometry") is None
                    or vp.get("address_v4") is None
                    or vp.get("asn_v4") is None
                    or vp.get("country_code") is None
                    or self.is_geoloc_disputed(vp)
                ):
                    rejected += 1
                    continue

                if ipV6 and vp.get("address_v6") is None:
                    continue

                reduced_vp = {
                    "address_v4": vp["address_v4"],
                    "asn_v4": vp["asn_v4"],
                    "country_code": vp["country_code"],
                    "geometry": vp["geometry"],
                    "lat": vp["geometry"]["coordinates"][1],
                    "lon": vp["geometry"]["coordinates"][0],
                    "id": vp["id"],
                    "is_anchor": vp["is_anchor"],
                }

                if ipV6:
                    reduced_vp = {
                        "address_v4": vp["address_v4"],
                        "address_v4": vp["address_v6"],
                        "asn_v4": vp["asn_v4"],
                        "asn_v4": vp["asn_v6"],
                        "country_code": vp["country_code"],
                        "geometry": vp["geometry"],
                        "lat": vp["geometry"]["coordinates"][1],
                        "lon": vp["geometry"]["coordinates"][0],
                        "id": vp["id"],
                        "is_anchor": vp["is_anchor"],
                    }

                vps.append(reduced_vp)

        logger.info(f"Retrieved {len(vps)} VPs connected on RIPE Atlas")
        logger.info(f"VPs removed: {rejected}")
        logger.info(
            f"Number of Probes  = {len([vp for vp in vps if not vp['is_anchor']])}"
        )
        logger.info(f"Number of Anchors = {len([vp for vp in vps if vp['is_anchor']])}")

        return vps

    def parse_vp(self, vp, asndb) -> str:
        """parse RIPE Atlas VPs data for clickhouse insertion"""

        _, bgp_prefix = route_view_bgp_prefix(vp["address_v4"], asndb)
        subnet = get_prefix_from_ip(vp["address_v4"])

        return f"""{vp['address_v4']},\
            {subnet},\
            {vp['asn_v4']},\
            {bgp_prefix},\
            {vp['country_code']},\
            {vp['lat']},\
            {vp['lon']},\
            {vp['id']},\
            {vp['is_anchor']}"""

    async def insert_vps(self, vps: list[dict], output_table: str) -> None:
        """insert vps within clickhouse db"""
        asndb = pyasn.pyasn(str(self.settings.RIB_TABLE))

        csv_data = []
        for vp in vps:
            parsed_vp = self.parse_vp(vp, asndb)
            csv_data.append(parsed_vp)

        # check if vps already exists
        prev_vps = None
        try:
            prev_vps = load_vps(output_table)
        except ClickHouseException:
            logger.debug("VPs table does not exists, proceeding with normal setup")

        tmp_file_path = create_tmp_csv_file(csv_data)

        async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
            if prev_vps:
                logger.warning(
                    f"VPs table:: {output_table} already exists, dropping it"
                )
                await DropTable().aio_execute(client, output_table)

            await CreateVPsTable().aio_execute(client, output_table)
            await InsertCSV().aio_execute(
                client=client,
                table_name=output_table,
                data=tmp_file_path.read_bytes(),
            )

        tmp_file_path.unlink()

    ###########################################################################################################
    # PING RELATED FUNCTIONS (RUN / PARSE)                                                                    #
    ###########################################################################################################
    async def ping(
        self,
        target: str,
        vp_ids: list[str],
        probing_tag: str,
        max_retry: int = 3,
        timeout: int = 60 * 5,
        wait_time: int = 60 * 5,
        protocol: str = "ICMP",
        ipv6: bool = False,
    ) -> int:
        """start ping measurement towards target from vps, return Atlas measurement id"""

        id = None
        async with httpx.AsyncClient() as client:
            for _ in range(max_retry):
                try:
                    resp = await client.post(
                        self.measurement_url,
                        headers=self.headers,
                        json=self.get_ping_config(
                            target, vp_ids, probing_tag, protocol, ipv6
                        ),
                        timeout=timeout,
                    )
                    resp = resp.json()
                except httpx.ReadTimeout as e:
                    logger.error(
                        "Read timeout for post request, retrying, max retry = 3"
                    )
                    logger.error(f"{e}")
                    await asyncio.sleep(wait_time)
                    continue
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Cannot parse response:: {e}, {resp}")
                    await asyncio.sleep(wait_time)
                    continue
                except Exception as e:
                    logger.error(f"Unsuported error:: {e}")
                    await asyncio.sleep(wait_time)
                    continue

                try:
                    id = resp["measurements"][0]
                    break
                except KeyError as e:
                    logger.error(f"{probing_tag}::STOPPED::Too many measurements!! {e}")
                    logger.error(f"{resp=}")
                    await asyncio.sleep(wait_time)
                    break
            else:
                raise Exception(
                    f"{probing_tag}:: Cannot perform measurement for target: {target}"
                )
        return id

    def parse_ping(self, results: list[dict], ipv6: bool = False) -> list[str]:
        """retrieve all measurement, parse data and return for clickhouse insert"""
        parsed_data = []
        for result in results:
            try:
                rtts = [rtt["rtt"] for rtt in result["result"] if "rtt" in rtt]
            except TypeError as e:
                logger.error(f"{e}:: {result=}")

            if not "from" in result or not "dst_addr" in result:
                continue

            if not result["from"] or not result["dst_addr"]:
                logger.error(
                    f"could not retrive ping {result['msm_id']}:: {result['from']=}, {result['dst_addr']=}"
                )
                continue

            if not rtts:
                rtts = [-1]

            parsed_data.append(
                f"{result['timestamp']},\
                {result['from']},\
                {get_prefix_from_ip(result['from'], ipv6)},\
                {24},\
                {result['prb_id']},\
                {result['msm_id']},\
                {result['dst_addr']},\
                {get_prefix_from_ip(result['dst_addr'], ipv6)},\
                {result['proto']},\
                {result['rcvd']},\
                {result['sent']},\
                {min(rtts)},\
                {max(rtts)},\
                {result['avg']},\
                \"{rtts}\""
            )

        return parsed_data

    ###########################################################################################################
    # TRACEROUTE RELATED FUNCTIONS (RUN / PARSE)                                                              #
    ###########################################################################################################
    async def traceroute(
        self,
        target: str,
        vp_ids: list[str],
        probing_tag: str,
        max_retry: int = 3,
        timeout: int = 60 * 5,
        wait_time: int = 60 * 5,
        min_ttl: int = 1,
        max_hops: int = 64,
        protocol: str = "ICMP",
        ipv6: bool = False,
    ) -> int:
        id = None
        async with httpx.AsyncClient() as client:
            for _ in range(max_retry):
                try:
                    req = self.get_traceroute_config(
                        target,
                        vp_ids,
                        probing_tag,
                        min_ttl,
                        max_hops,
                        protocol,
                        ipv6,
                    )
                    resp = await client.post(
                        self.measurement_url,
                        headers=self.headers,
                        json=req,
                        timeout=timeout,
                    )
                    resp = resp.json()
                except httpx.ReadTimeout as e:
                    logger.error(
                        "Read timeout for post request, retrying, max retry = 3"
                    )
                    logger.error(f"{e}")
                    await asyncio.sleep(wait_time)
                    continue
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Cannot parse response:: {e}, {resp}")
                    await asyncio.sleep(wait_time)
                    continue
                except Exception as e:
                    logger.error(f"Unsuported error:: {e}")
                    await asyncio.sleep(wait_time)
                    continue

                try:
                    id = resp["measurements"][0]
                    break
                except KeyError as e:
                    logger.error(f"{probing_tag}::STOPPED::Too many measurements!! {e}")
                    logger.error(f"{resp=}")
                    logger.error(f"{req=}")
                    await asyncio.sleep(wait_time)
                    break
            else:
                logger.error(
                    f"{probing_tag}:: Cannot perform measurement for target: {target}"
                )
        return id

    def parse_traceroute(self, traceroutes: list, ipv6: bool = False) -> str:
        """retrieve all measurement, parse data and return for clickhouse insert"""
        traceroute_results = []
        for traceroute in traceroutes:

            if not traceroute["from"] or not traceroute["dst_addr"]:
                continue

            for ttl_results in traceroute["result"]:
                ttl = ttl_results["hop"]

                # hop 255, junk data
                if ttl == 255:
                    continue

                # retrieve rtt from response
                rcvd = 0
                try:
                    sent = len(ttl_results["result"])
                except KeyError:
                    continue

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
                    try:
                        if not ipv6:
                            private_ip_addr = IPv4Address(ip_addr).is_private
                        else:
                            private_ip_addr = IPv6Address(ip_addr).is_private
                    except AddressValueError:
                        continue

                    if private_ip_addr:
                        continue

                    min_rtt = min(rtts)
                    max_rtt = max(rtts)
                    avg_rtt = mean(rtts)

                    traceroute_results.append(
                        f"{traceroute['timestamp']},\
                        {traceroute['from']},\
                        {get_prefix_from_ip(traceroute['from'], ipv6)},\
                        {24},\
                        {traceroute['prb_id']},\
                        {traceroute['msm_id']},\
                        {traceroute['dst_addr']},\
                        {get_prefix_from_ip(traceroute['dst_addr'], ipv6)},\
                        {traceroute['proto']},\
                        {ip_addr},\
                        {get_prefix_from_ip(ip_addr, ipv6)},\
                        {ttl},\
                        {sent},\
                        {rcvd},\
                        {min_rtt},\
                        {max_rtt},\
                        {avg_rtt},\
                        \"{rtts}\""
                    )

        return traceroute_results

    ###########################################################################################################
    # GET AND INSERT PING AND TRACEROUTES                                                                     #
    ###########################################################################################################
    async def get_measurement_results(
        self,
        id: int,
        measurement_type: str,
        timeout: int = 30,
        max_retry: int = 3,
        wait_time: int = 30,
        ipv6: bool = False,
    ) -> list[str]:
        """get results from ping measurement id"""
        url = f"{self.measurement_url}/{id}/results/"
        async with httpx.AsyncClient(timeout=timeout) as client:
            for i in range(max_retry):
                try:
                    resp = await client.get(url, headers=self.headers, timeout=timeout)
                    resp = resp.json()

                    measurement_results = []
                    match measurement_type:
                        case "ping":
                            measurement_results = self.parse_ping(resp, ipv6=ipv6)
                        case "traceroute":
                            measurement_results = self.parse_traceroute(resp)
                        case _:
                            raise RuntimeError(
                                f"Unknown {measurement_type=}; cannot retrieve measurement"
                            )

                    await asyncio.sleep(0.1)
                except httpx.ReadTimeout as e:
                    logger.error(f"{e}")
                    logger.error(f"Insertion pending, retry:: {i+1}/{max_retry}")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Unsuported error:: {e}")
                    await asyncio.sleep(wait_time)

        return measurement_results
