import json
import httpx
import asyncio

from numpy import mean
from ipaddress import IPv4Address
from collections import defaultdict
from pyasn import pyasn
from loguru import logger
from enum import Enum
from pych_client import AsyncClickHouseClient
from datetime import datetime, timedelta

from geogiant.clickhouse import CreateVPsTable, CreateTracerouteTable, Insert, Drop
from geogiant.common.files_utils import create_tmp_csv_file
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

    # TODO: decorator
    def check_schedule_validity(self, schedule: dict) -> None:
        """for any target, check if not too many VPs are scheduled"""
        for target, vp_ids in schedule.items():
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
                    "tags": [uuid],
                    "description": f"Dioptra Geolocation of {target}",
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

    def is_geoloc_disputed(self, probe: dict) -> bool:
        """check if geoloc disputed flag is contained in probe metadata"""
        tags = probe["tags"]
        for tag in tags:
            if tag["slug"] == "system-geoloc-disputed":
                return True
        return False

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

    def parse_vp(self, vp, asndb: pyasn) -> str:
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
        asndb = pyasn(str(self.settings.RIB_TABLE))

        csv_data = []
        for vp in vps:
            parsed_vp = self.parse_vp(vp, asndb)
            csv_data.append(parsed_vp)

        tmp_file_path = create_tmp_csv_file(csv_data)

        async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
            await CreateVPsTable().execute(client, self.settings.VPS_RAW)
            await Insert().execute(
                client=client,
                table_name=table_name,
                data=tmp_file_path.read_bytes(),
            )

        tmp_file_path.unlink()

    async def get_raw_traceroutes(
        self,
        type: str = "traceroute",
        af: int = 4,
        window: int = 1,
    ) -> dict:
        """get all measurement"""

        wd_start_time = datetime.now() - timedelta(days=window)
        wd_start_time = datetime.timestamp(wd_start_time)

        params = {
            "start_time__gte": wd_start_time,
            "type": type,
            "af": af,
            "status__in": RIPEAtlasStatus.STOPPED.value,
            "page_size": 100,
        }

        async with httpx.AsyncClient() as client:
            # get first results page
            resp = await client.get(self.measurement_url, params=params)
            resp = resp.json()

            # only get results for valid measurements
            for measurement in resp["results"]:
                if measurement["participant_count"]:
                    results_url = measurement["result"]
                    results = await client.get(results_url)
                    results = results.json()

                    for traceroute in results:
                        yield traceroute

            await asyncio.sleep(0.1)

            # iterate until no next page available
            i = 0
            while next_url := resp.get("next"):
                logger.debug(f"Next page::{resp['next']}")

                resp = await client.get(next_url)
                resp = resp.json()

                # only get results for valid measurements
                for measurement in resp["results"]:
                    if measurement["participant_count"]:
                        results_url = measurement["result"]
                        results = await client.get(results_url)
                        results = results.json()

                        for traceroute in results:
                            yield traceroute

                await asyncio.sleep(0.1)

                # TODO::REMOVE
                if i >= 2:
                    break
                i += 1

    def parse_traceroute(self, traceroute: dict) -> str:
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
                    {sent},\
                    {rcvd},\
                    {min_rtt},\
                    {max_rtt},\
                    {avg_rtt},\
                    \"{rtts}\""
                )

        return traceroute_results

    async def insert_traceroutes(
        self, batch_size: int = 10_000, drop_table: bool = False
    ) -> None:
        """retrieve traceroutes from RIPE Atlas and insert results in clickhouse"""
        traceroutes = []

        if drop_table:
            async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
                await Drop().execute(client, self.settings.TRACEROUTE_VPS_TO_TARGET)

        # drop table for testing
        async for traceroute in self.get_raw_traceroutes():
            parsed_traceroute = self.parse_traceroute(traceroute)

            if parsed_traceroute:
                traceroutes.extend(parsed_traceroute)

            # insert every buffer size to avoid losing data
            if len(traceroutes) > batch_size:
                logger.info(f"inserting batch of traceroute: limit = {batch_size}")

                # create table + insert
                tmp_file_path = create_tmp_csv_file(traceroutes)

                async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
                    await CreateTracerouteTable().execute(
                        client, self.settings.TRACEROUTE_VPS_TO_TARGET
                    )
                    await Insert().execute(
                        client=client,
                        table_name=self.settings.TRACEROUTE_VPS_TO_TARGET,
                        data=tmp_file_path.read_bytes(),
                    )

                tmp_file_path.unlink()
                traceroutes = []

    def parse_ping(self, results: list[dict]) -> list[str]:
        """retrieve all measurement, parse data and return for clickhouse insert"""
        parsed_data = []
        # TODO: check if loop necessary
        for result in results:
            rtts = [rtt["rtt"] for rtt in result["result"] if "rtt" in rtt]

            if rtts:
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
                    {result['avg']},\
                    {rtts}"
                )

        return parsed_data

    async def get_ping_results(self, id: int) -> dict:
        """get results from ping measurement id"""
        url = f"{self.measurement_url}/{id}/results/"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp = resp.json()

        # parse data and return to prober
        ping_results = self.parse_ping(resp)

        return ping_results

    async def get_status(self, measurement_id: str) -> bool:
        """check if measurement status is ongoing, if"""

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.measurement_url}/{measurement_id}/")
            resp = resp.json()

        if resp["status"]["name"] != "Ongoing":
            return True

        return False

    async def get_vps(self, probes_only: bool = False) -> list:
        """return all RIPE Atlas VPs (set probe_only to remove anchors)"""
        vps = []
        rejected = 0
        vp: dict = None
        i = 0
        async for vp in self.get_raw_vps():
            # TODO::REMOVE
            if i > 10:
                break
            i += 1

            # filter vps based on generic criteria
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

                # TODO: add query timestamp?
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

    async def request_stream(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict,
    ) -> dict:
        """return stream bytes for large files"""
        async with client.stream("GET", url, params=params) as resp:
            async for stream_data in resp.aiter_bytes():
                yield json.loads(stream_data.decode())

    async def ping(
        self,
        target: str,
        vp_ids: list[str],
        uuid: str,
        max_retry: int = 3,  # TODO: retry decorator (tenacity)?
    ) -> int:
        """start ping measurement towards target from vps, return Atlas measurement id"""

        id = None
        async with httpx.AsyncClient() as client:
            for _ in range(max_retry):
                resp = await client.post(
                    self.measurement_url
                    + f"/?key={self.settings.RIPE_ATLAS_SECRET_KEY}",
                    json=self.get_ping_config(target, vp_ids),
                )
                resp = resp.json()

                try:
                    id = resp["measurements"][0]
                    break
                except KeyError as e:
                    logger.error(f"{uuid}::STOPPED::Too many measurements!! {e}")
                    await asyncio.sleep(60)
            else:
                raise Exception(
                    f"{uuid}:: Cannot perform measurement for target: {target}"
                )
        return id


from datetime import date, datetime
from time import time
from ftplib import FTP
from argparse import ArgumentParser
from subprocess import call
from sys import stdout
from urllib.request import urlopen


# Parse command line options
# Note: --latest might be changes to --46, instead of --4, in the future
parser = ArgumentParser(
    description="Script to download MRT/RIB BGP archives (from RouteViews)."
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "--latestv4", "-4", "--latest", action="store_true", help="Grab lastest IPV4 data"
)
group.add_argument(
    "--latestv6", "-6", action="store_true", help="Grab lastest IPV6 data"
)
group.add_argument(
    "--latestv46", "-46", action="store_true", help="Grab lastest IPV4/V6 data"
)
group.add_argument("--version", action="store_true")
group.add_argument(
    "--dates-from-file",
    "-f",
    action="store",
    help="Grab IPV4 archives for specifc dates (one date, YYYYMMDD, per line)",
)
parser.add_argument(
    "--filename", action="store", help="Specify name with which the file will be saved"
)
args = parser.parse_args()


def ftp_download(server, remote_dir: str, remote_file: str, local_file):
    """Downloads a file from an FTP server and stores it locally"""
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(remote_dir)
    logger.info("Downloading ftp://%s/%s/%s" % (server, remote_dir, remote_file))
    filesize = ftp.size(remote_file)

    filename = args.filename if args.filename is not None else local_file
    # perhaps warn before overwriting file?
    with open(filename, "wb") as fp:

        def recv(s):
            fp.write(s)
            recv.chunk += 1
            recv.bytes += len(s)
            if recv.chunk % 100 == 0:
                print(
                    "\r %.f%%, %.fKB/s"
                    % (
                        recv.bytes * 100 / filesize,
                        recv.bytes / (1000 * (time() - recv.start)),
                    ),
                    end="",
                )
                stdout.flush()

        recv.chunk, recv.bytes, recv.start = 0, 0, time()
        ftp.retrbinary("RETR %s" % remote_file, recv)
    ftp.close()
    logger.info("\nDownload complete.")


def find_latest_in_ftp(server, archive_root, sub_dir, print_progress=True):
    """Returns (server, filepath, filename) for the most recent file in an FTP archive"""
    if print_progress:
        print("Connecting to ftp://" + server)
    ftp = FTP(server)
    ftp.login()
    months = sorted(
        ftp.nlst(archive_root), reverse=True
    )  # e.g. 'route-views6/bgpdata/2016.12'
    filepath = "/%s/%s" % (months[0], sub_dir)
    if print_progress:
        print("Finding most recent archive in %s ..." % filepath)
    ftp.cwd(filepath)
    fls = ftp.nlst()
    if not fls:
        filepath = "/%s/%s" % (months[1], sub_dir)
        if print_progress:
            print("Finding most recent archive in %s ..." % filepath)
        ftp.cwd(filepath)
        fls = ftp.nlst()
        if not fls:
            raise LookupError(
                "Cannot find file to download. Please report a bug on github?"
            )
    filename = max(fls)
    ftp.close()
    return (server, filepath, filename)
