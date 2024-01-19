import time
import asyncio

from pyasn import pyasn
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from copy import copy
from loguru import logger
from pych_client import AsyncClickHouseClient

from clickhouse import CreatePingTable, Insert
from ripe_atlas_api import RIPEAtlasAPI

from common.files_utils import dump_json, create_tmp_csv_file
from common.settings import ClickhouseSettings


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
