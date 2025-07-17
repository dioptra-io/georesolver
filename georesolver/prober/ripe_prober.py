import asyncio

from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from loguru import logger
from random import shuffle
from datetime import datetime, timedelta
from pych_client import ClickHouseClient

from georesolver.clickhouse import CreatePingTable, CreateTracerouteTable, InsertCSV

from georesolver.clickhouse.queries import get_measurement_ids
from georesolver.prober import RIPEAtlasAPI
from georesolver.common.files_utils import dump_json, create_tmp_csv_file
from georesolver.common.settings import ClickhouseSettings


class RIPEAtlasProber:
    """perform a measurement on RIPE Atlas and output data within Clickhouse"""

    def __init__(
        self,
        probing_type: str,
        probing_tag: str,
        output_table: str,
        uuid: str = None,
        dry_run: bool = False,
        output_logs: Path = None,
        min_ttl: int = 1,
        max_hops: int = 32,
        protocol: str = "icmp",
        ipv6: bool = False,
    ) -> None:
        self.dry_run = dry_run

        self.api = RIPEAtlasAPI()
        self.settings = ClickhouseSettings()

        self.probing_tag = probing_tag
        self.output_table = output_table
        self.uuid = uuid4() if not uuid else uuid
        self.probing_type = probing_type
        self.table_name = output_table + "__" + str(self.uuid)
        self.start_time = datetime.timestamp(datetime.now())
        self.end_time = None

        self.nb_ongoing_measurements: int = 0
        self.measurement_done: bool = False
        self.config: dict = None
        self.measurement_ids = []
        self.output_logs = output_logs
        self.ipv6 = ipv6

        # only for traceroute
        self.min_ttl = min_ttl
        self.max_hops = max_hops
        self.protocol = protocol

    async def init_prober(self) -> None:
        """get connected vps from measurement platform, insert in clickhouse"""
        vps = await self.api.get_vps()
        await self.api.insert_vps(vps, self.api.settings.VPS_RAW_TABLE)

    def get_config(self, schedule: dict) -> dict:
        """create a dictionary config for the new measurement"""
        return {
            "probing_type": self.probing_type,
            "probing_tag": str(self.probing_tag),
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
                tag=self.probing_tag
            )
            if self.nb_ongoing_measurements is None:
                self.nb_ongoing_measurements = self.api.settings.MAX_MEASUREMENT

            await asyncio.sleep(wait_time)

    async def insert_ids(self) -> None:
        """fetch periodically new measurements ids and put them into the config"""
        logger.info(f"{self.uuid} Saving all ping ids")

        # wait for the measurement to finish
        while not self.measurement_done:

            # dump measurement ids into config file
            self.config["ids"] = self.measurement_ids
            self.save_config(
                self.config, out_path=self.api.settings.MEASUREMENTS_CONFIG
            )

            await asyncio.sleep(5)

        # dump measurement ids into config file
        self.config["ids"] = self.measurement_ids
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
                    probing_tag=str(self.probing_tag),
                    protocol=self.protocol,
                    ipv6=self.ipv6,
                )
            case "traceroute":
                id = await self.api.traceroute(
                    target=target,
                    vp_ids=[vp_id for vp_id in vp_ids],
                    probing_tag=str(self.probing_tag),
                    min_ttl=self.min_ttl,
                    max_hops=self.max_hops,
                    protocol=self.protocol,
                    ipv6=self.ipv6,
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
                f"Starting measurement, slots available:: {self.api.settings.MAX_MEASUREMENT - self.nb_ongoing_measurements}"
            )

            id = await self.probe(
                target=target,
                vp_ids=vp_ids,
                uuid=self.uuid,
            )

            measurement_start_time = datetime.timestamp(datetime.now())

            self.measurement_ids.append((id, measurement_start_time))

            # so we do not have to wait for actualization
            self.nb_ongoing_measurements += 1
            await asyncio.sleep(wait_time)

        self.measurement_done = True
        self.end_time = str(datetime.now())

        logger.info(f"{self.uuid}::Measurement finished")

    async def insert_measurement(
        self,
        ids: list[int],
        output_table: str,
        wait_time: int = 0.1,
        step_size: int = 1,
        ipv6: bool = False,
    ) -> None:
        """insert a specific type of measurement"""
        csv_data = []
        for i in tqdm(range(0, len(ids), step_size)):
            tasks = [
                self.api.get_measurement_results(id, self.probing_type, ipv6=ipv6)
                for id in ids[i : i + step_size]
            ]
            measurement_results = await asyncio.gather(*tasks)

            for r in measurement_results:
                csv_data.extend(r)

            await asyncio.sleep(wait_time)

        with ClickHouseClient(**self.settings.clickhouse) as client:
            tmp_file_path = create_tmp_csv_file(csv_data)
            match self.probing_type:
                case "ping":
                    CreatePingTable().execute(client, output_table)
                case "traceroute":
                    CreateTracerouteTable().execute(client, output_table)
                case _:
                    raise RuntimeError(
                        f"Insert measurement:: Unknown measurement type {self.probing_type}"
                    )

            InsertCSV().execute(
                client=client, table_name=output_table, data=tmp_file_path.read_bytes()
            )
            tmp_file_path.unlink()

    async def insert_measurements(
        self,
        measurement_schedule: list[tuple],
        wait_time: int = 60,
    ) -> None:
        """insert measurement once they are tagged as Finished on RIPE Atlas"""
        current_time = datetime.timestamp(datetime.now() - timedelta(days=2))
        cached_measurement_ids = set()
        while True:
            # load measurement finished from RIPE Atlas
            stopped_measurement_ids = await RIPEAtlasAPI().get_stopped_measurement_ids(
                start_time=current_time, tags=["dioptra", self.probing_tag]
            )

            # load already inserted measurement ids
            inserted_ids = get_measurement_ids(self.output_table)

            # stop measurement once all measurement are inserted
            all_measurement_ids = set(inserted_ids).union(cached_measurement_ids)
            if len(all_measurement_ids) >= len(measurement_schedule):
                logger.info(
                    f"All measurement inserted:: {len(inserted_ids)=}; {len(measurement_schedule)=}"
                )
                break

            measurement_to_insert = set(stopped_measurement_ids).difference(
                set(inserted_ids)
            )

            # check cached measurements,
            # some measurement are not insersed because no results
            measurement_to_insert = set(measurement_to_insert).difference(
                cached_measurement_ids
            )

            logger.info(f"{len(stopped_measurement_ids)=}")
            logger.info(f"{len(inserted_ids)=}")
            logger.info(f"{len(measurement_to_insert)=}")

            if not measurement_to_insert:
                await asyncio.sleep(wait_time)
                continue

            # insert measurement
            batch_size = 10
            for i in range(0, len(measurement_to_insert), batch_size):
                logger.info(
                    f"Batch {i // batch_size}/{len(measurement_to_insert) // batch_size}"
                )
                batch_measurement_ids = list(measurement_to_insert)[i : i + batch_size]
                await self.insert_measurement(batch_measurement_ids, self.output_table)

            cached_measurement_ids.update(measurement_to_insert)
            current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

            await asyncio.sleep(wait_time)

    async def main(
        self, schedule: dict, config: dict = None, with_insert: bool = True
    ) -> Path:
        """run measurement schedule using RIPE Atlas API"""

        # check if schedule is in accordance with API's parameters
        self.api.check_schedule_validity(schedule)
        shuffle(schedule)

        self.measurement_done = False

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

        if with_insert:
            await asyncio.gather(
                self.run(schedule),
                self.insert_ids(),
                self.insert_measurements(schedule),
                self.ongoing_measurements(),
            )
        else:
            await asyncio.gather(
                self.run(schedule),
                self.insert_ids(),
                self.ongoing_measurements(),
            )

        # condition to stop Insert results function
        self.config["start_time"] = self.start_time
        self.config["end_time"] = self.end_time

        output_path = self.save_config(
            self.config, out_path=self.api.settings.MEASUREMENTS_CONFIG
        )

        logger.info(f"Measurement done, output path:: {output_path}")

        return output_path
