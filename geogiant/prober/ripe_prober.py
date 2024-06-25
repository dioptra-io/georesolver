import time
import asyncio

from random import shuffle
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from copy import copy
from loguru import logger

from geogiant.prober import RIPEAtlasAPI

from geogiant.common.files_utils import dump_json
from geogiant.common.settings import ClickhouseSettings


class RIPEAtlasProber:
    """perform a measurement on RIPE Atlas and output data within Clickhouse"""

    def __init__(
        self,
        probing_type: str,
        probing_tag: str,
        uuid: str = None,
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run

        self.api = RIPEAtlasAPI()
        self.settings = ClickhouseSettings()

        self.uuid = uuid4() if not uuid else uuid
        self.probing_type = probing_type
        self.table_name = probing_tag + "__" + str(self.uuid)
        self.start_time = time.time()
        self.end_time = None

        self.measurement_done = False
        self.config: dict = None
        self.measurement_ids = set()

    async def init_prober(self) -> None:
        """get connected vps from measurement platform, insert in clickhouse"""
        vps = await self.api.get_vps()
        await self.api.insert_vps(vps, self.api.settings.VPS_RAW)

    def get_config(self, schedule: dict) -> dict:
        """create a dictionary config for the new measurement"""
        return {
            "probing_type": self.probing_type,
            "uuid": str(self.uuid),
            "status": "ongoing",
            "start_time": str(datetime.now()),
            "end_time": None,
            "is_dry_run": self.dry_run,
            "nb_targets": len(schedule),
            "af": self.api.settings.IP_VERSION,
            "ids": None,
        }

    def save_config(self, config: dict, out_path: Path) -> None:
        """save newly created measurement configuration"""

        if config["end_time"] is not None:
            config["status"] = "finished"

        config_file = out_path / f"{self.table_name}.json"
        dump_json(config, config_file)

        return config_file

    async def wait_for_batch(self, ongoing_ids: set) -> list:
        """Wait for a measurement batch to end"""

        logger.info(f"Concurrent Number of measurement limit reached")
        logger.info(f"Nb concurrent measurement: {len(ongoing_ids)}")
        logger.info(
            f"Waiting for measurement batch to end before starting a new one..."
        )

        # allows that some measurement might be stuck
        finished_counter = 0
        while finished_counter < self.api.settings.MAX_MEASUREMENT:
            tmp_ids = copy(ongoing_ids)

            for id in tmp_ids:
                finished = await self.api.get_status(id)
                if finished:
                    logger.debug(f"removing id:: {id} from ongoing measurements")
                    ongoing_ids.remove(id)
                    finished_counter += 1

                await asyncio.sleep(0.1)
            await asyncio.sleep(60)

        return ongoing_ids

    async def probe(self, target: str, vp_ids: list, uuid: str) -> dict:
        """Ping one target IP address from a list of vp ids"""
        logger.info(
            f"{self.uuid}::starting measurement for {target} from {len(vp_ids)} vps"
        )

        match (self.probing_type):
            case "ping":
                id = await self.api.ping(
                    target=target,
                    vp_ids=[vp_id for vp_id in vp_ids],
                    uuid=str(uuid),
                )
            case "traceroute":
                id = await self.api.traceroute(
                    target=target,
                    vp_ids=[vp_id for vp_id in vp_ids],
                    uuid=str(uuid),
                )
            case _:
                raise RuntimeError(
                    f"Measurement type:: {self.probing_type} not supported"
                )

        logger.info(f"{target=}, {id=}")
        return id

    async def run(self, schedule: list[tuple]) -> list[int]:
        """run measurement schedule"""
        ongoing_ids = set()
        for target, vp_ids in schedule:
            id = await self.probe(
                target=target,
                vp_ids=vp_ids,
                uuid=self.uuid,
            )

            self.measurement_ids.add(id)
            ongoing_ids.add(id)

            # wait for a batch of measurement to finish
            if len(ongoing_ids) >= self.api.settings.MAX_MEASUREMENT:
                ongoing_ids = await self.wait_for_batch(ongoing_ids)

            # wait some time between the beginning of each measurements
            time.sleep(0.1)

        self.measurement_done = True
        self.end_time = time.time()

        logger.info(f"{self.uuid}::Measurement finished")

    async def insert(self, config: dict) -> None:
        """fetch periodically new measurements"""
        logger.info(f"{self.uuid} Saving all ping ids")

        # wait for the measurement to finish
        inserted_ids = set()
        ids_to_insert = None
        while not self.measurement_done and not ids_to_insert:
            # get measurement that were not already inserted
            ids_to_insert = inserted_ids.symmetric_difference(self.measurement_ids)

            if ids_to_insert:
                # dump measurement ids into config file
                config["ids"] = list(self.measurement_ids)
                self.save_config(config, out_path=self.api.settings.MEASUREMENTS_CONFIG)
                inserted_ids.update(ids_to_insert)
                logger.debug("Measurement Ids saved")
                ids_to_insert = None

            await asyncio.sleep(60 * 5)

        # get measurement that were not already inserted
        ids_to_insert = inserted_ids.symmetric_difference(self.measurement_ids)

        if ids_to_insert:
            # dump measurement ids into config file
            config["ids"] = list(self.measurement_ids)
            self.save_config(config, out_path=self.api.settings.MEASUREMENTS_CONFIG)
            inserted_ids.update(ids_to_insert)
            logger.debug("Measurement Ids saved")
            ids_to_insert = None

    async def main(self, schedule: dict, config: dict = None) -> Path:
        """run measurement schedule using RIPE Atlas API"""

        # check if schedule is in accordance with API's parameters
        self.api.check_schedule_validity(schedule)
        shuffle(schedule)

        dump_json(
            data=schedule,
            output_file=self.api.settings.MEASUREMENTS_SCHEDULE
            / f"{self.table_name}.json",
        )

        if not config:
            config = self.get_config(schedule)
            self.save_config(config, out_path=self.api.settings.MEASUREMENTS_CONFIG)

        logger.info(
            f"{self.uuid}::Starting measurement for {len({t[0] for t in schedule})} targets"
        )

        await asyncio.gather(self.run(schedule), self.insert(config))

        # condition to stop Insert results function
        self.measurement_done = True
        config["ids"] = list(self.measurement_ids)
        config["start_time"] = self.start_time
        config["end_time"] = self.end_time

        output_path = self.save_config(
            config, out_path=self.api.settings.MEASUREMENTS_CONFIG
        )

        return output_path
