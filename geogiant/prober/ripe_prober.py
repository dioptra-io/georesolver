import time
import asyncio

from random import shuffle
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from copy import copy
from loguru import logger
from tqdm import tqdm

from geogiant.prober import RIPEAtlasAPI
from geogiant.common.queries import insert_pings, insert_traceroutes
from geogiant.common.files_utils import dump_json
from geogiant.common.settings import ClickhouseSettings


class RIPEAtlasProber:
    """perform a measurement on RIPE Atlas and output data within Clickhouse"""

    def __init__(
        self,
        probing_type: str,
        probing_tag: str,
        output_table: str,
        uuid: str = None,
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run

        self.api = RIPEAtlasAPI()
        self.settings = ClickhouseSettings()

        self.output_table = output_table
        self.uuid = uuid4() if not uuid else uuid
        self.probing_type = probing_type
        self.table_name = probing_tag + "__" + str(self.uuid)
        self.start_time = datetime.timestamp(datetime.now())
        self.end_time = None

        self.nb_ongoing_measurements: int = 0
        self.measurement_done: bool = False
        self.config: dict = None
        self.measurement_ids = set()

    async def init_prober(self) -> None:
        """get connected vps from measurement platform, insert in clickhouse"""
        vps = await self.api.get_vps()
        await self.api.insert_vps(vps, self.api.settings.VPS_RAW_TABLE)

    def get_config(self, schedule: dict) -> dict:
        """create a dictionary config for the new measurement"""
        return {
            "probing_type": self.probing_type,
            "uuid": str(self.uuid),
            "status": "ongoing",
            "start_time": str(datetime.timestamp(datetime.now())),
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

    async def ongoing_measurements(self, wait_time: int = 5) -> None:
        """simply check how many measurements have the status Ongoing"""
        while not self.measurement_done:
            self.nb_ongoing_measurements = await self.api.get_ongoing_measurements(
                tags=["dioptra"]
            )
            if self.nb_ongoing_measurements is None:
                self.nb_ongoing_measurements = self.api.settings.MAX_MEASUREMENT

            await asyncio.sleep(wait_time)

    async def retrieve_pings_from_tag(self, tag, output_table: str) -> None:
        """retrieve all results for a specific tag and insert data"""
        csv_data = await RIPEAtlasAPI().get_measurements_from_tag(tag)

        await insert_pings(csv_data, output_table)

    async def retrieve_pings(
        self, ids: list[int], output_table: str, wait_time: int = 0.1
    ) -> None:
        """retrieve all ping measurements from a list of measurement ids"""
        csv_data = []
        for id in tqdm(ids):
            ping_results = await RIPEAtlasAPI().get_ping_results(id)
            csv_data.extend(ping_results)

            await asyncio.sleep(wait_time)

        await insert_pings(csv_data, output_table)

    async def retrieve_traceroutes(
        self,
        ids: list[int],
        output_table: str,
        wait_time: float = 0.1,
    ) -> list[dict]:
        """retrieve all traceroutes from a list of ids"""
        csv_data = []
        for id in tqdm(ids):
            traceroute_result = await RIPEAtlasAPI().get_traceroute_results(id)
            csv_data.extend(traceroute_result)

            await asyncio.sleep(wait_time)

        await insert_traceroutes(csv_data, output_table)

    async def insert_results(self, wait_time: int = 5) -> None:
        """insert ongoing measurements"""
        inserted_measurements = set()
        current_time = self.start_time
        while not self.measurement_done:

            # 1. get all stopped measurements
            stopped_measurement_ids = await self.api.get_stopped_measurement_ids(
                tags=["dioptra"],
                start_time=current_time,
            )

            # 2. filter already inserted measurement ids
            current_measurement_ids = self.measurement_ids

            # 3. only keep ids related with this measurement
            stopped_measurement_ids = set(stopped_measurement_ids).intersection(
                set(current_measurement_ids)
            )

            if not stopped_measurement_ids:
                await asyncio.sleep(wait_time)
                continue

            measurement_to_insert = set(stopped_measurement_ids).difference(
                set(inserted_measurements)
            )

            if not measurement_to_insert:
                continue

            logger.info(f"Measurement to insert:: {len(measurement_to_insert)}")

            if self.probing_type == "ping":
                await self.retrieve_pings(
                    measurement_to_insert, self.output_table, wait_time=1
                )
            elif self.probing_type == "traceroute":
                await self.retrieve_traceroutes(
                    measurement_to_insert, self.output_table, wait_time=1
                )
            else:
                raise RuntimeError(f"{self.probing_type} not supported")

            inserted_measurements.update(measurement_to_insert)

            current_time = datetime.timestamp(datetime.now())

            await asyncio.sleep(wait_time)

        # check if there are still some measurement to insert
        current_measurement_ids = self.config["ids"]
        stopped_measurement_ids = await self.api.get_stopped_measurement_ids(
            tags=["dioptra"],
            start_time=current_time,
        )

        # 2. filter already inserted measurement ids
        current_measurement_ids = self.config["ids"]

        # 3. only keep ids related with this measurement
        stopped_measurement_ids = set(stopped_measurement_ids).difference(
            set(current_measurement_ids)
        )

        if stopped_measurement_ids:
            measurement_to_insert = set(stopped_measurement_ids).difference(
                set(inserted_measurements)
            )

            logger.info(f"Measurement to insert:: {len(measurement_to_insert)}")

            if measurement_to_insert:

                if self.probing_type == "pings":
                    await self.retrieve_pings(measurement_to_insert, self.output_table)
                elif self.probing_type == "traceroutes":
                    await self.retrieve_traceroutes(
                        measurement_to_insert, self.output_table
                    )
                else:
                    raise RuntimeError(f"{self.probing_type} not supported")

                inserted_measurements.update(measurement_to_insert)

        logger.info("All measurements were inserted")

    async def insert_ids(self) -> None:
        """fetch periodically new measurements ids and put them into the config"""
        logger.info(f"{self.uuid} Saving all ping ids")

        # wait for the measurement to finish
        while not self.measurement_done:

            # dump measurement ids into config file
            self.config["ids"] = list(self.measurement_ids)
            self.save_config(
                self.config, out_path=self.api.settings.MEASUREMENTS_CONFIG
            )

            await asyncio.sleep(5)

        # dump measurement ids into config file
        self.config["ids"] = list(self.measurement_ids)
        self.save_config(self.config, out_path=self.api.settings.MEASUREMENTS_CONFIG)

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

    async def run(self, schedule: list[tuple], wait_time: int = 0.1) -> list[int]:
        """run measurement schedule"""
        for target, vp_ids in schedule:

            # check if a new measurement can be started or not
            print_debug = True
            while self.nb_ongoing_measurements >= self.api.settings.MAX_MEASUREMENT:
                if print_debug:
                    logger.info(
                        f"API limit reached, waiting for measurements to finish. {self.nb_ongoing_measurements=}"
                    )
                await asyncio.sleep(1)
                print_debug = False

            print_debug = True
            logger.info(
                f"Restarting measurements, slots available:: {self.api.settings.MAX_MEASUREMENT - self.nb_ongoing_measurements}"
            )

            id = await self.probe(
                target=target,
                vp_ids=vp_ids,
                uuid=self.uuid,
            )

            self.measurement_ids.add(id)

            # so we do not have to wait for actualization
            self.nb_ongoing_measurements += 1
            await asyncio.sleep(wait_time)

        self.measurement_done = True
        self.end_time = str(datetime.now())

        logger.info(f"{self.uuid}::Measurement finished")

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
            self.config = self.get_config(schedule)
            self.save_config(
                self.config, out_path=self.api.settings.MEASUREMENTS_CONFIG
            )
        else:
            self.config = config

        logger.info(
            f"{self.uuid}::Starting measurement for {len({t[0] for t in schedule})} targets"
        )

        await asyncio.gather(
            self.run(schedule),
            self.insert_ids(),
            self.ongoing_measurements(),
            self.insert_results(),
        )

        # condition to stop Insert results function
        self.config["start_time"] = self.start_time
        self.config["end_time"] = self.end_time

        output_path = self.save_config(
            self.config, out_path=self.api.settings.MEASUREMENTS_CONFIG
        )

        logger.info(f"Measurement done, output path:: {output_path}")

        return output_path
