"""general functions for controling"""

import json
import pickle
import bz2

from tqdm import tqdm
from ipaddress import IPv4Address, AddressValueError
from typing import Generator
from typing import Iterator
from uuid import uuid4
from loguru import logger
from dateutil import parser
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


def decompress(bz2_file: Path) -> None:
    """decompress bz2 archive into the same directory"""
    logger.info(f"Decompressing:: {bz2_file}")
    output_file_path = bz2_file.parent / (str(bz2_file.name).split(".bz2")[0])
    with open(output_file_path, "wb") as new_file, bz2.BZ2File(bz2_file, "rb") as file:
        for data in iter(lambda: file.read(100 * 1024), b""):
            new_file.write(data)

    logger.info(f"{bz2_file} sucessfully decompressed")

    return output_file_path


def load_iter(in_file: Path) -> Generator:
    """load iter large file"""
    for row in in_file.open("r"):
        try:
            yield json.loads(row)
        except:
            yield None


def load_json_iter(file_path: Path) -> Generator:
    """iter load json file"""
    with file_path.open("r") as f:
        for row in f.readlines():
            yield json.loads(row)


def iter_file(file: str, *, read_size: int = 2**20) -> Iterator[bytes]:
    with open(file, "rb") as f:
        while True:
            chunk = f.read(read_size)
            if not chunk:
                break
            yield chunk


def dump_csv(data: list[str], output_file: Path) -> None:
    """output data into output file with json format"""
    # check that dir exists before writing
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        for row in data:
            f.write(row + "\n")


def load_csv(input_file: Path) -> list[str]:
    """load csv file, return list of string"""
    data = []
    try:
        with input_file.open("r") as f:
            for row in f.readlines():
                data.append(row.strip("\n"))
    except FileNotFoundError as e:
        logger.error(f"file does not exists: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"the file you are trying to retrieve is empty: {e}")

    return data


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


def create_tmp_csv_file(csv_data: list[str]) -> Path:
    """create a tmp file into TMP with uuid"""
    file_path = path_settings.TMP_PATH / f"tmp__{uuid4()}.csv"

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w") as f:
        for row in csv_data:
            f.write(row + "\n")

    return file_path


def create_tmp_json_file(json_data) -> Path:
    """create a tmp file into TMP with uuid"""
    file_path = path_settings.TMP_PATH / f"tmp__{uuid4()}.csv"

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    dump_json(json_data, file_path)

    return file_path


def load_pickle(input_file: Path) -> dict:
    """load json file"""
    try:
        with input_file.open("rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"could not load json file: {e}")


def load_countries_info() -> dict:
    """load all countries info"""
    countries = {}
    with path_settings.COUNTRIES_INFO.open("r") as f:
        for row in f.readlines()[1:]:
            row = row.split(",")
            continent = row[1]
            alpha2_country_code = row[2]
            default_lat = row[5]
            default_lon = row[6]

            try:
                countries[alpha2_country_code] = {
                    "default_lat": float(default_lat),
                    "default_lon": float(default_lon),
                    "continent": continent,
                }
            except IndexError or ValueError:
                continue

    return countries


def load_countries_default_geoloc() -> dict:
    """load countries default geolocation"""
    countries = {}
    with (path_settings.DATASET / "countries_default_geo.txt").open("r") as f:
        for row in f.readlines():
            row = [value.strip() for value in row.split(" ")]

            try:
                countries[row[0]] = {
                    "lat": float(row[1]),
                    "lon": float(row[2]),
                    "code": row[3],
                }
            except IndexError or ValueError:
                continue

    return countries


def load_countries_continent() -> dict:
    """return a dict of each countries continent"""
    countries_continent = {}
    with path_settings.COUNTRIES_INFO.open("r") as f:
        for row in f.readlines()[1:]:
            row = row.split(",")
            continent = row[1]
            country_code = row[2]

            countries_continent[country_code] = continent

    return countries_continent


def load_anycatch_data() -> None:
    """get all anycast prefixes detected by anycatch and remove them"""
    anycast_prefixes = set()
    with path_settings.ANYCATCH_DATA.open("r") as f:
        for row in f.readlines():
            prefix = row.split(",")[0].strip("\n")
            anycast_prefixes.add(prefix)

    return anycast_prefixes


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


def generate_routers_targets_file(routers_path: Path) -> None:
    if not (routers_path).exists():
        rows = load_json_iter(path_settings.DATASET / "routers.json")
        targets = {}
        for row in rows:
            addr = row["ip"]
            # remove IPv6 and private IP addresses
            try:
                if not IPv4Address(addr).is_private:
                    subnet = get_prefix_from_ip(addr)
                    targets[addr] = {
                        "subnet": subnet,
                        "lat": row["probe_latitude"],
                        "lon": row["probe_longitude"],
                    }
            except AddressValueError:
                continue

        logger.info(f"Number of targets in routers datasets:: {len(targets)}")

        dump_json(targets, routers_path)


def ip_addresses_hitlist() -> None:
    targets_per_prefix = defaultdict(list)

    total_addr = set()
    total_subnets = set()
    with open(path_settings.ADDRESS_FILE, "r") as f:
        for row in tqdm(f.readlines()[1:]):
            row = row.split("\t")

            try:
                score = row[1]
            except IndexError:
                continue

            if int(score) < 50:
                continue

            ip_addr = row[-1].strip("\n")
            subnet = get_prefix_from_ip(ip_addr)

            targets_per_prefix[subnet].append(ip_addr)

            total_addr.add(ip_addr)
            total_subnets.add(subnet)

    logger.info(f"{len(total_addr)=}")
    logger.info(f"{len(total_subnets)=}")

    dump_json(targets_per_prefix, path_settings.USER_HITLIST_FILE)
