import asyncio


from random import shuffle
from uuid import uuid4
from pyasn import pyasn
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from geogiant.prober import RIPEAtlasAPI
from geogiant.common.queries import (
    get_measurement_ids,
    insert_pings,
    insert_traceroutes,
    load_target_geoloc,
    insert_geoloc,
    load_geoloc,
    load_vps,
)
from geogiant.common.ip_addresses_utils import route_view_bgp_prefix, get_prefix_from_ip
from geogiant.common.files_utils import dump_json, load_json
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
        output_logs: Path = None,
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

    async def watch_measurements(
        self,
        ongoing_measurements: list[int],
        measurement_timeout: int = 60 * 15,
    ) -> list[int]:
        """check if ongoing measurements are lasting longer than defined timeout"""
        measurement_stopped = []
        current_time = datetime.now()
        for id, start_time in self.measurement_ids:
            if not id in ongoing_measurements:
                continue

            start_time = datetime.fromtimestamp(start_time)
            if start_time + timedelta(seconds=measurement_timeout) > current_time:
                logger.info(f"Measurement {self.uuid}:: measurement {id} timed out")
                stopped_id = self.api.stop_measurement(id)
                measurement_stopped.append(stopped_id)

        return measurement_stopped

    async def retrieve_pings_from_tag(self, tag, output_table: str) -> None:
        """retrieve all results for a specific tag and insert data"""
        csv_data = await RIPEAtlasAPI().get_measurements_from_tag(tag)

        await insert_pings(csv_data, output_table)

    async def retrieve_pings(
        self, ids: list[int], output_table: str, wait_time: int = 0.1
    ) -> None:
        """retrieve all ping measurements from a list of measurement ids"""
        csv_data = []

        if self.output_logs:
            output_file = self.output_logs.open("a")
        else:
            output_file = None

        for id in tqdm(ids, file=output_file):
            ping_results = await RIPEAtlasAPI().get_ping_results(id)
            csv_data.extend(ping_results)

            await asyncio.sleep(wait_time)

        if output_file:
            output_file.close()

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

    def parse_geoloc_data(self, target_geoloc: dict) -> list[str]:
        """parse ping data to geoloc csv"""
        csv_data = []
        asndb = pyasn(str(self.api.settings.RIB_TABLE))
        vps = load_vps(self.settings.VPS_RAW_TABLE)

        vps_per_vps_addr = {}
        for vp in vps:
            vps_per_vps_addr[vp["addr"]] = vp

        for target_addr, shortest_ping_data in target_geoloc.items():
            target_subnet = get_prefix_from_ip(target_addr)
            target_asn, target_bgp_prefix = route_view_bgp_prefix(target_addr, asndb)
            if not target_bgp_prefix or not target_asn:
                target_bgp_prefix = "Unknown"
                target_asn = -1

            msm_id = shortest_ping_data[1]
            min_rtt = shortest_ping_data[2]

            # filter unresponisive targets
            if min_rtt == -1:
                continue

            vp_addr = shortest_ping_data[0]
            try:
                vp = vps_per_vps_addr[vp_addr]
            except KeyError:
                logger.debug(f"VP {vp_addr} does not exists")
                continue

            csv_data.append(
                f"{target_addr},\
                {target_subnet},\
                {target_bgp_prefix},\
                {target_asn},\
                {vp['lat']},\
                {vp['lon']},\
                {vp['country_code']},\
                {vp_addr},\
                {vp['subnet']},\
                {vp['bgp_prefix']},\
                {vp['asn_v4']},\
                {min_rtt},\
                {msm_id}"
            )

        return csv_data

    async def insert_geoloc_from_pings(
        self, ping_table: str, geoloc_table: str
    ) -> None:
        """insert all geoloc in clickhouse"""
        target_geoloc = load_target_geoloc(table_name=ping_table)
        csv_data = self.parse_geoloc_data(target_geoloc)

        await insert_geoloc(
            csv_data=csv_data,
            output_table=geoloc_table,
        )

    async def insert_results(
        self, nb_targets: int, geoloc_table: str, wait_time: int = 60
    ) -> None:
        """insert ongoing measurements"""
        inserted_measurements = get_measurement_ids(self.output_table)
        current_time = self.start_time
        insert_done = False
        while not insert_done:

            # Get all stopped measurements
            stopped_measurement_ids = await self.api.get_stopped_measurement_ids(
                start_time=current_time,
                tags=["dioptra", self.probing_tag],
            )

            # Find stopped measurements that are not inserted yet
            measurement_to_insert = set(stopped_measurement_ids).difference(
                set(inserted_measurements)
            )

            if not measurement_to_insert:
                logger.info(f"{self.probing_tag} :: No measurement to insert")
                await asyncio.sleep(wait_time)
                continue

            logger.info(f"Measurement            :: {self.probing_tag}")
            logger.info(f"Measurements done      :: {len(stopped_measurement_ids)}")
            logger.info(f"Measurements to insert :: {len(measurement_to_insert)}")

            if self.probing_type == "ping":
                await self.retrieve_pings(
                    measurement_to_insert, self.output_table, wait_time=1
                )

                await self.insert_geoloc_from_pings(
                    ping_table=self.output_table,
                    geoloc_table=geoloc_table,
                )

            elif self.probing_type == "traceroute":
                await self.retrieve_traceroutes(
                    measurement_to_insert, self.output_table, wait_time=1
                )
            else:
                raise RuntimeError(f"{self.probing_type} not supported")

            current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

            inserted_measurements.update(measurement_to_insert)

            # stopping point for the insertion
            if len(inserted_measurements) >= nb_targets:
                insert_done = True

            await asyncio.sleep(wait_time)

        logger.info("All measurements inserted")

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
                )
            case "traceroute":
                id = await self.api.traceroute(
                    target=target,
                    vp_ids=[vp_id for vp_id in vp_ids],
                    probing_tag=str(self.probing_tag),
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

    async def main(self, schedule: dict, config: dict = None) -> Path:
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
