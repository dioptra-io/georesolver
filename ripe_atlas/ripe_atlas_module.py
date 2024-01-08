import json
import time
import requests

from datetime import datetime
from pathlib import Path
from copy import copy
from loguru import logger

from common.ip_addresses_utils import get_prefix_from_ip
from common.files_utils import dump_json

from insert_results import InsertPingMeasurement
from settings import RIPEAtlasSettings

class RIPEAtlas:
    def __init__(
        self,
        schedule: dict,
        output_table: str,
        username: str = None,
        secret_key: str = None,
        tag: str = None,
        ip_version: int = 4,
        dry_run: bool = False,
    ) -> None:
        self.schedule = schedule
        self.output_table = output_table
        self.check_schedule_validity(self.schedule)
        
        # static parameters
        self.tag = tag
        self.dry_run = dry_run
        
        # user credentials override
        self.settings = RIPEAtlasSettings()
        if username and secret_key:
            self.settings.USERNAME = username
            self.settings.SECRET_KEY = secret_key
        if ip_version:
            self.settings.IP_VERSION = ip_version
            
    def check_schedule_validity(self) -> None:
        """for any target, check if not too many VPs are scheduled"""
        for target, vp_ids in self.schedule.items():
            if len(vp_ids) > self.settings.MAX_VP:
                raise RuntimeError(f"Too many VPs scheduled for target: {target} (nb vps: {len(vp_ids)}, max: {self.settings.MAX_VP})")
    
    def get_response(url: str, max_retry: int = 10, wait_time: int = 2) -> list:
        """request to Atlas API"""

        for _ in range(max_retry):
            response = requests.get(url)

            # small parsing, as response might not be Json formatted
            try:
                response = json.loads(response.content)
            except json.JSONDecodeError:
                response = response.content.decode()
                response = response.replace("}{", "}, {")
                response = response.replace("} {", "}, {")
                response = json.loads(response)

            if response != []:
                break
            
            time.sleep(wait_time)

        return response
    
    def get_measurements_from_tag(self, tag: str) -> dict:
        """retrieve all measurements that share the same tag and return parsed measurement results"""

        url = f"{self.settings.MEASUREMENT_URL}/tags/{tag}/results/"

        response = self.get_response(url, max_retry=1, wait_time=1)

        return response

    def save_config(config: dict, out_path: Path) -> None:
        """save newly created measurement configuration"""

        if config["end_time"] is not None:
            config["status"] = "finished"

        config_file = out_path / f"{config['measurement_tag']}.json"
        dump_json(config, config_file)

    def get_measurement_config(
        self
    ) -> dict:
        """create a dictionary config for the new measurement"""
        return {
            "tag": str(self.tag),
            "status": "ongoing",
            "start_time": str(datetime.now()),
            "end_time": None,
            "is_dry_run": self.dry_run,
            "nb_targets": len(self.schedule),
            "af": self.settings.IP_VERSION,
            "ids": None,
        }

    
    def get_json_config(
        self, 
        target: str,
        vp_ids: list[str],
        nb_packets: int = None) -> dict:
        return {
            "definitions": [
                {
                    "target": target,
                    "af": self.settings.IP_VERSION,
                    "packets": nb_packets if nb_packets else self.settings.PING_NB_PACKETS,
                    "size": 48,
                    "tags": [self.tag],
                    "description": f"Dioptra Geolocation of {target}",
                    "resolve_on_probe": False,
                    "skip_dns_check": True,
                    "include_probe_id": False,
                    "type": "ping",
                }
            ],
            "probes": [
                {"value": v_id, "type": "probes", "requested": 1}
                for v_id in vp_ids
            ],
            "is_oneoff": True,
            "bill_to": self.settings.USERNAME,
        }

    def ping(
        self,
        target: str,
        vp_ids: list[str],
        max_retry: int = 3,
        nb_packets: int = None,
    ) -> None:
        """start ping measurement towards target from vps, return Atlas measurement id"""

        measurement_id = None
        for _ in range(max_retry):
            json_config = self.get_json_config(target,vp_ids,nb_packets)
            
            response = requests.post(
                self.settings.MEASUREMENT_URL,
                json=json_config,
            ).json()

            try:
                measurement_id = response["measurements"][0]
                break
            except KeyError as e:
                logger.info(f"error: {e}")
                logger.warning("Too many measurements Waiting.")
                time.sleep(60)
        else:
            raise Exception("Cannot post ping measurement")

        return measurement_id


    def wait_for(self, measurement_id: str, max_retry: int = 30) -> None:
        for _ in range(max_retry):
            response = requests.get(
                f"{self.settings.MEASUREMENT_URL}/{measurement_id}/"
            ).json()

            # check if measurement is ongoing or not
            if response["status"]["name"] != "Ongoing":
                return response

            time.sleep(10)

        return None
    
    def wait_for_measurement_slot(
        self,
        active_ids: list,
    ) -> list:
        """check if a measurement can be started, hold until measurement stack is free

        Args:
            active_ids (list): the list of ongoing measurements

        Return:
            list: active measurement list
        """
        if (
            len(active_ids)
            >= self.settings.MAX_MEASUREMENT
        ):
            logger.info(f"Wait for starting new measurement")
            logger.info(f"limit is: {len(active_ids)}")

            tmp_measurement_ids = copy(active_ids)
            for id in tmp_measurement_ids:
                # wait for the last measurement of the batch to end before starting a new one
                measurement_result = self.wait_for(id)
                if measurement_result:
                    active_ids.remove(id)

        return active_ids
    
    def query(
        self,
        target: str,
        vp_ids: list,
        active_ids : list,
        all_ids : list,
    ) -> dict:
        """
        Ping one target from a list of vp ids
        """
        ripe_atlas_driver = RIPEAtlas()
        
        logger.info(
            f"starting measurement for {target} from {len(vp_ids)} vps"
        )

        id = ripe_atlas_driver.ping(
            target=target,
            vp_ids=[vp_id for vp_id in vp_ids],
            nb_packets=self.settings.PING_NB_PACKETS,
        )

        active_ids.append(id)
        all_ids.append(id)

        logger.info(
            f"measurement tag: {self.tag} : started measurement id : {id}"
        )

        active_ids = self.wait_for_measurement_slot(active_ids)
        
        return active_ids, all_ids

    def get_measurement_from_id(self,id: int,
    ) -> dict:
        """retrieve measurement results from RIPE Atlas with measurement id"""
        url = f"{self.settings.MEASUREMENT_URL}/{id}/results"
        response = self.get_response(url)

        return response
    
    def parse_data(self, config: dict) -> list[str]:
        """retrieve all measurement, parse data and return for clickhouse insert"""
        for id in config["measurement_ids"]:
            # get measurement results
            results = self.get_measurement_from_id(id)
            
            # check if measurement belongs to target or responsive subnet address
            ping_data = []
            for result in results:
                rtts = [rtt["rtt"] for rtt in result["result"] if "rtt" in rtt]

                if rtts:
                    ping_data.append(f"{result['timestamp']},\
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
                        {rtts}")
                    
            # do not overload RIPE Atlas api
            time.sleep(0.1)
        
    def run(
        self,
        config: dict = None,
    ) -> None:
        # check that no measurement config was given as input
        if not config:
            config = self.get_measurement_config()
            self.save_config(config)

        # perform measurement
        active_ids = []
        all_ids = []
        start_time = time.time()
        for target, vp_ids in self.schedule.items():
            
            active_ids, all_ids = self.query(
                target=target,
                vp_ids=vp_ids,
                active_ids=active_ids,
                all_ids=all_ids
            )
            
        logger.info(f"Measurement : {self.tag} done")
        
        end_time = time.time()
        config["ids"] = all_ids
        config["start_time"] = start_time
        config["end_time"] = end_time

        self.save_config(
            config, out_path=self.settings.MEASUREMENT_CONFIG_PATH
        )
        
        logger.info("Waiting for several minutes for probes to upload results...")
        time.sleep(60 * 10)
        logger.info("Getting measurement results")
        
        csv_data = self.parse_data(config)
        
        # finally insert results
        InsertPingMeasurement().insert(
            output_table=self.output_table,
            input_data=csv_data,
        )
           