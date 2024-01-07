"""general functions for controling"""
import json
import pickle

from loguru import logger
from dateutil import parser
from datetime import datetime
from uuid import UUID
from pathlib import Path



def dump_json(data: dict, output_file: Path) -> None:
    """output data into output file with json format"""
    # check that dir exists before writing
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(data, f, indent=4)


def insert_json(data, output_file: Path) -> None:
    """insert new dict data to existing json file"""
    # check that dir exists before writing
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    pre_existing_data = {}
    try:
        with output_file.open("r") as f:
            try:
                pre_existing_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"file {output_file} empty, cannot insert")
    except FileNotFoundError:
        logger.warning(f"trying to insert a non-existing file: {output_file}")

    # update data
    if type(data) == list:
        data.extend(pre_existing_data)
    if type(data) == dict:
        data.update(pre_existing_data)

    with output_file.open("w") as f:
        json.dump(data, f, indent=4)


def load_json(input_file: Path) -> dict:
    """load json file"""
    try:
        with input_file.open("r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logger.error(f"file does not exists: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"the file you are trying to retrieve is empty: {e}")

    return None


def dump_pickle(data: dict, output_file: Path) -> None:
    """output data into output file with json format"""
    # check that dir exists before writing
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file: Path) -> dict:
    """load json file"""
    try:
        with input_file.open("rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"could not load json file: {e}")


def get_measurement_config(
    measurement_tag: UUID,
    measurement_schedule: list,
    dry_run=False,
) -> dict:
    """return measurement config for future retrieval"""
    return {
        "measurement_tag": str(measurement_tag),
        "status": "ongoing",
        "start_time": str(datetime.now()),
        "end_time": None,
        "is_dry_run": dry_run,
        "nb_targets": len(measurement_schedule),
        "description": "ecs dns resolution anchors mapping",
        "af": 4,
        "measurement_ids": None,
        "measurement_schedule": measurement_schedule,
    }


def save_measurement_config(measurement_config: dict, out_path: Path) -> None:
    """save measurement config"""

    if measurement_config["end_time"] is not None:
        measurement_config["status"] = "finished"

    config_file = out_path / f"{measurement_config['measurement_tag']}.json"
    dump_json(measurement_config, config_file)


def get_latest_measurement_config(config_path: Path) -> dict:
    """retrieve latest measurement config"""
    try:
        assert config_path.is_dir()
    except AssertionError:
        logger.error(f"config path is not a dir: {config_path}")

    latest: datetime = None
    for file in config_path.iterdir():
        measurement_config = load_json(file)
        if latest:
            if latest < parser.isoparse(measurement_config["start_time"]):
                latest_config = measurement_config
        else:
            latest = parser.isoparse(measurement_config["start_time"])
            latest_config = measurement_config

    return latest_config


def get_measurement_id_from_logs(log_path: Path) -> list:
    """from measurement output logs, retrieve ripe measurement ids"""
    measurement_ids = []
    with open(log_path, "r") as f:
        measurements_counter = 0
        for row in f.readlines():
            if "started measurement id :" in row:
                id = row.split("started measurement id :")[-1].strip("\n")
                measurement_ids.append(id)
                measurements_counter += 1

        logger.info(f"nb measurements performed so far: {measurements_counter}")

    return measurement_ids


def match_mapping_to_vp(ripe_servers: list, dns_mapping_results: list) -> None:
    """from a set of dns mapping result and ripe servers description, return target/vp pairs"""
    for result in dns_mapping_results:
        ip_addr = result["subnet"].split("/24")[0]

        # find the corresponding server
        for server in ripe_servers:
            ip_server = server["address_v4"]

            if ip_server == ip_addr:
                server["cdn_mapping"] = [
                    {
                        "hostname": result["answers"]["hostname"],
                        "answers": result["answers"]["answers"],
                    }
                ]

    return ripe_servers
