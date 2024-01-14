import time
import httpx
import asyncio

from pyasn import pyasn
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from copy import copy
from loguru import logger
from pych_client import AsyncClickHouseClient

from clickhouse import CreateVPsTable, CreatePingTable, Insert

from common.files_utils import dump_json, create_tmp_csv_file
from common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from common.settings import RIPEAtlasSettings, ClickhouseSettings


class RIPEAtlasAPI:
    """RIPE Atlas measurement API"""

    settings = RIPEAtlasSettings()

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
            "bill_to": self.settings.USERNAME,
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
            resp: dict = resp.json()

            for vp in resp["results"]:
                yield vp

            await asyncio.sleep(0.1)

            i = 0
            while url := resp.get("next"):
                logger.debug(f"Next page::{resp['next']}")
                resp = await client.get(resp["next"])
                resp = resp.json()

                for vp in resp["results"]:
                    yield vp

                await asyncio.sleep(0.1)

    async def insert_vps(self, vps: list[dict], table_name: str) -> None:
        """insert vps within clickhouse db"""
        asndb = pyasn(str(self.settings.RIB_TABLE))

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

        tmp_file_path = create_tmp_csv_file(csv_data)

        async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
            await CreateVPsTable().execute(client, self.settings.VPS_RAW)
            await Insert().execute(
                client=client,
                table_name=table_name,
                data=tmp_file_path.read_bytes(),
            )

        tmp_file_path.unlink()

    def parse_ping(self, results: list[dict], id: int) -> list[str]:
        """retrieve all measurement, parse data and return for clickhouse insert"""
        parsed_data = []
        for result in results:
            rtts = [rtt["rtt"] for rtt in result["result"] if "rtt" in rtt]

            if rtts:
                parsed_data.append(
                    f"{result['timestamp']},\
                    {result['from']},\
                    {get_prefix_from_ip(result['from'])},\
                    {24},\
                    {result['prb_id']},\
                    {id},\
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
        url = f"{self.settings.MEASUREMENT_URL}/{id}/results/"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp = resp.json()

        # parse data and return to prober
        ping_results = self.parse_ping(resp, id)

        return ping_results

    async def get_status(self, measurement_id: str) -> bool:
        """check if measurement status is ongoing, if"""

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.settings.MEASUREMENT_URL}/{measurement_id}/"
            )
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
                    self.settings.MEASUREMENT_URL,
                    json=self.get_ping_config(target, vp_ids),
                )
                resp = resp.json()

                try:
                    id = resp["measurements"][0]
                    break
                except KeyError as e:
                    logger.warning(f"{uuid}::STOPPED::Too many measurements!! {e}")
                    await asyncio.sleep(60)
            else:
                raise Exception(
                    f"{uuid}:: Cannot perform measurement for target: {target}"
                )

        return id


class RIPEAtlasProber:
    """perform a measurement on RIPE Atlas and output data within Clickhouse"""

    def __init__(
        self,
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run

        self.api = RIPEAtlasAPI()
        self.settings = ClickhouseSettings()

        self.uuid = uuid4()
        self.table_name = "ping__" + str(self.uuid)
        self.start_time = time.time()

        self.measurement_done = False
        self.config: dict = None
        self.measurement_ids = set()

    async def init_prober(self) -> None:
        """get connected vps from measurement platform, insert in clickhouse"""
        vps = await self.api.get_vps()
        await self.api.insert_vps(vps, self.api.settings.VPS_RAW)

    def get_config(self, schedule) -> dict:
        """create a dictionary config for the new measurement"""
        return {
            "uuid": str(self.uuid),
            "status": "ongoing",
            "start_time": str(datetime.now()),
            "end_time": None,
            "is_dry_run": self.dry_run,
            "nb_targets": len(schedule),
            "af": self.api.settings.IP_VERSION,
            "ids": None,
        }

    def save_config(slef, config: dict, out_path: Path) -> None:
        """save newly created measurement configuration"""

        if config["end_time"] is not None:
            config["status"] = "finished"

        config_file = out_path / f"{config['measurement_uuid']}.json"
        dump_json(config, config_file)

    async def wait_for_batch(self, ongoing_ids: set) -> list:
        """Wait for a measurement batch to end"""

        logger.info(f"Concurrent Number of measurement limit reached")
        logger.info(f"Max concurrent measurement: {len(ongoing_ids)}")
        logger.info(
            f"Waiting for measurement batch to end before starting a new one..."
        )

        # allows that some measurement might be stuck
        finished_counter = 0
        while finished_counter < self.api.settings.MAX_MEASUREMENT - 10:
            tmp_ids = copy(ongoing_ids)

            for id in tmp_ids:
                finished = await self.api.get_status(id)
                if finished:
                    ongoing_ids.remove(id)
                    finished_counter += 1

                await asyncio.sleep(0.1)
            await asyncio.sleep(60)

        return ongoing_ids

    async def probe(self, target: str, vp_ids: list) -> dict:
        """Ping one target IP address from a list of vp ids"""
        logger.info(
            f"{self.uuid}::starting measurement for {target} from {len(vp_ids)} vps"
        )

        id = await self.api.ping(
            target=target,
            vp_ids=[vp_id for vp_id in vp_ids],
            nb_packets=self.api.settings.PING_NB_PACKETS,
        )

        return id

    async def run(self, schedule: dict) -> list[int]:
        """run measurement schedule"""
        ongoing_ids = set()
        for target, vp_ids in schedule.items():
            id = await self.probe(
                target=target,
                vp_ids=vp_ids,
            )

            self.measurement_ids.add(id)
            ongoing_ids.add(id)

            # wait for a batch of measurement to finish
            if len(ongoing_ids) >= self.api.settings.MAX_MEASUREMENT:
                ongoing_ids = await self.wait_for_batch(ongoing_ids)

        self.measurement_done = True
        self.end_time = time.time()

        logger.info(f"{self.uuid}::Measurement finished")

    async def insert(self) -> None:
        """fetch periodically new measurements"""

        logger.info(f"{self.uuid}::Wait for first batch of measurement to be done")
        await asyncio.sleep(60 * 10)
        logger.info(f"{self.uuid}Getting measurement results")

        # wait for the measurement to finish
        inserted_ids = set()
        while not self.measurement_done and ids_to_insert:
            # get measurement that were not already inserted
            ids_to_insert = inserted_ids.symmetric_difference(self.measurement_ids)

            if ids_to_insert:
                csv_data = []
                for id in ids_to_insert:
                    ping_results = await self.api.get_ping_results(id)
                    csv_data.append(ping_results)

                    await asyncio.sleep(0.1)

                tmp_file_path = create_tmp_csv_file(csv_data)

                async with AsyncClickHouseClient(**self.settings.clickhouse) as client:
                    await CreatePingTable().execute(self.table_name)
                    await Insert().execute(
                        client=client,
                        table_name=self.table_name,
                        input_data=tmp_file_path.read_bytes(),
                    )

                tmp_file_path.unlink()

                ids_to_insert = None

    async def main(self, schedule: dict, config: dict = None) -> None:
        """run measurement schedule using RIPE Atlas API"""

        # check if schedule is in accordance with API's parameters
        self.api.check_schedule_validity(schedule)

        # TODO: add caching | replace with a DB?
        if not config:
            config = self.get_config(schedule)
            self.save_config(config)

        logger.info(
            f"{self.uuid}::Starting measurement for {len(schedule.values())} targets"
        )

        await asyncio.gather(
            self.run(schedule),
            self.insert(),
        )

        # condition to stop Insert results function
        self.measurement_done = True
        config["ids"] = list(self.measurement_ids)
        config["start_time"] = self.start_time
        config["end_time"] = self.end_time

        self.save_config(config, out_path=self.api.settings.MEASUREMENTS_CONFIG)


if __name__ == "__main__":
    target_addr = [""]
    vp_ids = [""]
