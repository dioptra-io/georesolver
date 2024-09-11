import httpx
import asyncio
import pyasn
import time
import json

from tqdm import tqdm
from numpy import mean
from ipaddress import IPv4Address, AddressValueError
from collections import defaultdict

from uuid import uuid4
from loguru import logger
from enum import Enum
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import (
    CreateVPsTable,
    CreatePingTable,
    InsertCSV,
    GetMeasurementIds,
)
from geogiant.common.files_utils import create_tmp_csv_file, dump_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import RIPEAtlasSettings


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

    def check_schedule_validity(self, schedule: list[tuple]) -> None:
        """for any target, check if not too many VPs are scheduled"""
        for target, vp_ids in schedule:
            if len(vp_ids) > self.settings.MAX_VP:
                raise RuntimeError(
                    f"Too many VPs scheduled for target: {target} (nb vps: {len(vp_ids)}, max: {self.settings.MAX_VP})"
                )

    def get_ping_config(self, target: str, vp_ids: list[int], uuid: str) -> dict:
        return {
            "definitions": [
                {
                    "target": target,
                    "af": self.settings.IP_VERSION,
                    "packets": self.settings.PING_NB_PACKETS,
                    "size": 48,
                    "tags": ["dioptra"],
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
            "bill_to": self.settings.RIPE_ATLAS_USERNAME,
        }

    def get_traceroute_config(self, target: str, vp_ids: list[int], uuid: str) -> dict:
        return {
            "definitions": [
                {
                    "target": target,
                    "af": self.settings.IP_VERSION,
                    "packets": self.settings.PING_NB_PACKETS,
                    "protocol": self.settings.PROTOCOL,
                    "tags": [uuid],
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
            # "bill_to": self.settings.RIPE_ATLAS_USERNAME,
        }

    def is_geoloc_disputed(self, probe: dict) -> bool:
        """check if geoloc disputed flag is contained in probe metadata"""
        tags = probe["tags"]
        for tag in tags:
            if tag["slug"] == "system-geoloc-disputed":
                return True
        return False

    async def get_ongoing_measurements(self, tags: list[str]) -> int:
        """return the number of measurements which have the status Ongoing"""
        async with httpx.AsyncClient() as client:
            params = {
                "sort": ["start_time"],
                "status__in": "Specified,Scheduled,Ongoing",
                "tags": tags,
                "mine": True,
                "key": self.settings.RIPE_ATLAS_SECRET_KEY,
            }
            resp = await client.get(self.measurement_url, params=params)
            resp = resp.json()

        try:
            ongoing_measurements = resp["count"]
        except KeyError:
            raise RuntimeError("Count key should be present in response")

        return ongoing_measurements

    async def get_stopped_measurement_ids(
        self, start_time: str, tags: list[str]
    ) -> list[int]:
        """return all measurements with a status stopped"""

        params = {
            "sort": ["start_time"],
            "status__in": "Stopped,Forced to stop,No suitable probes,Failed",
            "stop_time__gte": start_time,
            "tags": tags,
            "mine": True,
            "key": self.settings.RIPE_ATLAS_SECRET_KEY,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.get(self.measurement_url, params=params)
            resp = resp.json()

        stopped_measurements = [m["id"] for m in resp["results"]]

        while resp["next"]:
            async with httpx.AsyncClient() as client:
                resp = await client.get(resp["next"], params=params)
                resp = resp.json()

            stopped_measurements.extend([m["id"] for m in resp["results"]])

        return stopped_measurements

    async def get_raw_vps(self, url: str = "https://atlas.ripe.net/api/v2/probes/"):
        """get request url atlas endpoint"""
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
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

    async def insert_vps(self, vps: list[dict], table_name: str) -> None:
        """insert vps within clickhouse db"""
        asndb = pyasn.pyasn(str(self.settings.RIB_TABLE))

        csv_data = []
        for vp in vps:
            parsed_vp = self.parse_vp(vp, asndb)
            csv_data.append(parsed_vp)

        tmp_file_path = create_tmp_csv_file(csv_data)

        async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
            await CreateVPsTable().execute(client, self.settings.VPS_RAW)
            await InsertCSV().execute(
                client=client,
                table_name=table_name,
                data=tmp_file_path.read_bytes(),
            )

        tmp_file_path.unlink()

    async def insert_pings(self, ping_results: list[str], table_name: str) -> None:
        """insert vps within clickhouse db"""

        tmp_file_path = create_tmp_csv_file(ping_results)

        async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
            await CreatePingTable().aio_execute(client, table_name)
            await InsertCSV().aio_execute(
                client=client,
                table_name=table_name,
                data=tmp_file_path.read_bytes(),
            )

        tmp_file_path.unlink()

    async def get_ping_measurement_ids(self, table_name: str) -> list[int]:
        """return all measurement ids that were already inserted within clickhouse"""
        async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
            await CreatePingTable().aio_execute(client, table_name)
            resp = await GetMeasurementIds().aio_execute(client, table_name)

            measurement_ids = []
            for row in resp:
                measurement_ids.append(row["msm_id"])

            return measurement_ids

    def ripe_ip_map_locate(
        self, ip_addr: int, params: dict = {"engine": "single-radius"}
    ) -> int:
        ripe_ip_map_url = f"https://ipmap-api.ripe.net/v1/locate/{ip_addr}/best"
        ripe_ip_map_url += f"/?key={self.settings.RIPE_ATLAS_SECRET_KEY}"

        with httpx.Client() as client:
            resp = client.get(url=ripe_ip_map_url, params=params)
            resp = resp.json()

            contributions = resp["metadata"]["service"]["contributions"]
            for _, metadata in contributions.items():
                engines = metadata["engines"]

            for engine in engines:
                if engine["engine"] == "single-radius":
                    logger.info(f"{engine}")
                    try:
                        msm_id = engine["metadata"]["msmId"]
                        logger.info(f"Locate measurement id: {msm_id}")
                    except KeyError:
                        return None

        return msm_id

    def get_tag_measurement_ids(self, params: dict) -> None:
        # Define API endpoint and parameters
        url = self.api_url + f"/measurements/ping/"

        measurement_ids = []
        target_ip_addresses = set()
        with httpx.Client() as client:
            resp = client.get(url=url, params=params, timeout=15)
            measurements = resp.json()

            nb_pages = measurements["count"]
            logger.info(f"Loading {nb_pages} pages from RIPE Altas")

            count = 0
            if not measurements["next"]:
                for measurement in measurements["results"]:

                    logger.info(f"Page {count + 1}/{nb_pages}")

                    for measurement in measurements["results"]:
                        if (
                            measurement["af"] != 4
                            or measurement["target_ip"] in target_ip_addresses
                        ):
                            continue
                        logger.debug(measurement["id"])
                        measurement_ids.append(measurement["id"])
                        target_ip_addresses.add(measurement["target_ip"])

            while measurements["next"]:
                logger.info(f"Page {count + 1}/{nb_pages}")

                for measurement in measurements["results"]:
                    if (
                        measurement["af"] != 4
                        or measurement["target_ip"] in target_ip_addresses
                    ):
                        continue

                    measurement_ids.append(measurement["id"])
                    target_ip_addresses.add(measurement["target_ip"])

                logger.info("Loading next page")
                resp = client.get(measurements["next"], timeout=15)
                measurements = resp.json()
                count += 1

                time.sleep(1)

                if len(measurement_ids) > 100_000:
                    break

            return measurement_ids

    async def get_results_from_tag(self, params: dict, ping_table: str) -> None:
        # Define API endpoint and parameters
        url = self.api_url + f"/measurements/ping/"

        measurement_ids = []
        ping_results = []
        target_ip_addresses = set()

        ping_measurement_ids = await RIPEAtlasAPI().get_ping_measurement_ids(ping_table)

        with httpx.Client() as client:
            resp = client.get(url=url, params=params, timeout=15)
            measurements = resp.json()

            nb_pages = measurements["count"]
            logger.info(f"Loading {nb_pages} pages from RIPE Altas")

            count = 0
            while measurements["next"]:
                logger.info(f"Page {count + 1}/{nb_pages}")

                for measurement in measurements["results"]:
                    if (
                        measurement["af"] != 4
                        or measurement["target_ip"] in target_ip_addresses
                        or measurement["id"] in ping_measurement_ids
                    ):
                        continue

                    ping_result = self.get_ping_results(measurement["id"])

                    logger.info(f"{measurement['id']=}, {len(ping_result)}")

                    if len(ping_result) <= 1:
                        continue

                    ping_results.extend(ping_result)

                    await asyncio.sleep(0.5)

                    if len(ping_results) > 100:
                        await self.insert_pings(ping_results, table_name=ping_table)
                        ping_results = []

                logger.info("Loading next page")
                resp = client.get(measurements["next"], timeout=15)
                measurements = resp.json()
                count += 1

                await asyncio.sleep(1)

            return measurement_ids

    async def get_measurements_from_tag(self, tag: str):
        url = self.api_url + f"/measurements/tags/{tag}/results/"

        params = {"tags": tag, "format": "json"}
        ping_results = []
        with httpx.stream("GET", url=url, params=params, timeout=60) as client:
            decoded_bytes = b""
            for resp in client.iter_bytes():
                decoded_bytes += resp

        decoded_bytes = decoded_bytes.decode()

        dump_pickle(
            decoded_bytes, self.settings.END_TO_END_DATASET / "decoded_bytes.pickle"
        )

        ping_results = json.loads(resp)

        return self.parse_ping(ping_results)

    async def get_ping_results(self, id: int) -> dict:
        """get results from ping measurement id"""
        url = (
            f"{self.measurement_url}/{id}/results/"
            + f"/?key={self.settings.RIPE_ATLAS_SECRET_KEY}"
        )
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=30)
            resp = resp.json()

            ping_results = self.parse_ping(resp)
            await asyncio.sleep(0.1)

        return ping_results

    def get_probe_requested(self, id: int) -> int:
        url = (
            f"{self.measurement_url}/{id}/"
            + f"/?key={self.settings.RIPE_ATLAS_SECRET_KEY}"
        )
        with httpx.Client() as client:
            resp = client.get(url, timeout=30)
            resp = resp.json()

            time.sleep(0.1)

        return resp["probes_requested"], resp["target_ip"]

    def parse_ping(self, results: list[dict]) -> list[str]:
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
                {get_prefix_from_ip(result['from'])},\
                {24},\
                {result['prb_id']},\
                {result['msm_id']},\
                {result['dst_addr']},\
                {get_prefix_from_ip(result['dst_addr'])},\
                {result['proto']},\
                {result['rcvd']},\
                {result['sent']},\
                {min(rtts)},\
                {max(rtts)},\
                {result['avg']},\
                \"{rtts}\""
            )

        return parsed_data

    async def get_traceroute_info(self, id: int) -> dict:
        """get all measurement"""
        url = f"{self.measurement_url}/{id}/"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp = resp.json()

        return resp

    async def get_traceroute_results(self, id: int) -> dict:
        """get all measurement"""
        url = f"{self.measurement_url}/{id}/results/"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=15)
            resp = resp.json()

        traceroute_results = self.parse_traceroute(resp)

        return traceroute_results

    def parse_traceroute(self, traceroutes: list) -> str:
        """retrieve all measurement, parse data and return for clickhouse insert"""
        traceroute_results = []
        for traceroute in traceroutes:
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
                        private_ip_addr = IPv4Address(ip_addr).is_private
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
                        {sent},\
                        {rcvd},\
                        {min_rtt},\
                        {max_rtt},\
                        {avg_rtt},\
                        \"{rtts}\""
                    )

        return traceroute_results

    async def get_status(self, measurement_id: str) -> bool:
        """check if measurement status is ongoing, if"""

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.measurement_url}/{measurement_id}/")
            resp = resp.json()

        if resp["status"]["name"] not in [
            "Ongoing",
            "Scheduled",
            "Specified",
            "synchronizing",
        ]:
            logger.debug(f"{resp['status']['name']=}")
            return True

        return False

    async def get_vps(self, probes_only: bool = False) -> list:
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
                vps.append(reduced_vp)

        logger.info(f"Retrieved {len(vps)} VPs connected on RIPE Atlas")
        logger.info(f"VPs removed: {rejected}")
        logger.info(
            f"Number of Probes  = {len([vp for vp in vps if not vp['is_anchor']])}"
        )
        logger.info(f"Number of Anchors = {len([vp for vp in vps if vp['is_anchor']])}")

        return vps

    async def ping(
        self,
        target: str,
        vp_ids: list[str],
        uuid: str,
        max_retry: int = 3,
        timeout: int = 60 * 5,
        wait_time: int = 60,
    ) -> int:
        """start ping measurement towards target from vps, return Atlas measurement id"""

        id = None
        async with httpx.AsyncClient() as client:
            for _ in range(max_retry):
                try:
                    resp = await client.post(
                        self.measurement_url
                        + f"/?key={self.settings.RIPE_ATLAS_SECRET_KEY}",
                        json=self.get_ping_config(target, vp_ids, uuid),
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
                    logger.info(f"Cannot parse response:: {e}, {resp}")
                    await asyncio.sleep(wait_time)
                    continue

                try:
                    id = resp["measurements"][0]
                    break
                except KeyError as e:
                    logger.error(f"{uuid}::STOPPED::Too many measurements!! {e}")
                    logger.error(f"{resp=}")
                    await asyncio.sleep(wait_time)
                    break
            else:
                raise Exception(
                    f"{uuid}:: Cannot perform measurement for target: {target}"
                )
        return id

    async def traceroute(
        self, target: str, vp_ids: list[str], uuid: str, max_retry: int = 3
    ) -> int:
        id = None
        async with httpx.AsyncClient(timeout=60) as client:
            for _ in range(max_retry):
                resp = await client.post(
                    self.measurement_url
                    + f"/?key={self.settings.RIPE_ATLAS_SECRET_KEY}",
                    json=self.get_traceroute_config(target, vp_ids, uuid),
                    timeout=60,
                )

                resp = resp.json()

                try:
                    id = resp["measurements"][0]
                    break
                except KeyError as e:
                    logger.error(f"{uuid}::STOPPED::Too many measurements!! {e}")
                    logger.error(f"{resp=}")
                    await asyncio.sleep(60)
            else:
                raise Exception(
                    f"{uuid}:: Cannot perform measurement for target: {target}"
                )
        return id


async def test() -> None:
    """run one ping and one traceroute for testing"""
    target = "145.220.0.55"
    vps = [1136]

    id = await RIPEAtlasAPI().ping(target=target, vp_ids=vps, uuid=str(uuid4()))
    logger.info(f"Ping with measurement id:: {id} started")

    id = await RIPEAtlasAPI().traceroute(target=target, vp_ids=vps, uuid=str(uuid4()))
    logger.info(f"Traceroute with measurement id:: {id} started")


if __name__ == "__main__":

    asyncio.run(test())
