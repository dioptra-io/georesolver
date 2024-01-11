import time
import httpx
import asyncio

from uuid import uuid4
from datetime import datetime
from pathlib import Path
from copy import copy
from loguru import logger

from geogiant.clickhouse import PingTable

from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.files_utils import dump_json
from geogiant.common.settings import RIPEAtlasSettings


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

    async def get_ping_results(self, id: int) -> dict:
        """get results from ping measurement id"""
        url = f"{self.settings.MEASUREMENT_URL}/{id}/results/"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp = resp.json()

        return resp

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
        async for vp in self.get_raw_vps():
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
        schedule: dict,
        dry_run: bool = False,
    ) -> None:
        self.api = RIPEAtlasAPI()

        self.dry_run = dry_run
        self.schedule = schedule
        self.api.check_schedule_validity(self.schedule)

        self.uuid = uuid4()
        self.table_name = "ping__" + str(self.uuid)
        self.start_time = time.time()

        self.measurement_done = False
        self.config: dict = None
        self.measurement_ids = set()

    def get_config(self) -> dict:
        """create a dictionary config for the new measurement"""
        return {
            "uuid": str(self.uuid),
            "status": "ongoing",
            "start_time": str(datetime.now()),
            "end_time": None,
            "is_dry_run": self.dry_run,
            "nb_targets": len(self.schedule),
            "af": self.api.settings.IP_VERSION,
            "ids": None,
        }

    def save_config(slef, config: dict, out_path: Path) -> None:
        """save newly created measurement configuration"""

        if config["end_time"] is not None:
            config["status"] = "finished"

        config_file = out_path / f"{config['measurement_uuid']}.json"
        dump_json(config, config_file)

    def parse(self, results: list[dict], id: int) -> list[str]:
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

    async def run(self) -> list[int]:
        """run measurement schedule"""
        ongoing_ids = set()
        for target, vp_ids in self.schedule.items():
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
                    results = await self.api.get_ping_results(id)
                    parsed_data = self.parse(results, id)
                    csv_data.append(parsed_data)

                    await asyncio.sleep(0.1)

                PingTable().insert(
                    table_name=self.table_name,
                    input_data=csv_data,
                )

                ids_to_insert = None

    async def main(self, config: dict = None) -> None:
        """run measurement schedule using RIPE Atlas API"""

        # TODO: add caching | replace with a DB?
        if not config:
            config = self.get_config()
            self.save_config(config)

        logger.info(
            f"{self.uuid}::Starting measurement for {len(self.schedule.values())} targets"
        )

        await asyncio.gather(
            self.run(),
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
