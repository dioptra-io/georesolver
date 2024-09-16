"""VPs initialization functions"""

from tqdm import tqdm
from loguru import logger
from datetime import datetime
from collections import defaultdict
from pych_client import AsyncClickHouseClient
from pych_client.exceptions import ClickHouseException

from geogiant.prober import RIPEAtlasAPI, RIPEAtlasProber
from geogiant.clickhouse import CreateVPsTable, InsertFromCSV
from geogiant.common.queries import (
    load_vps,
    load_targets,
    get_vps_ids,
    get_pings_per_src_dst,
    get_pings_per_target,
)
from geogiant.common.geoloc import (
    distance,
    haversine,
    compute_remove_wrongly_geolocated_probes,
)
from geogiant.common.files_utils import (
    load_countries_info,
    dump_json,
    load_json,
    create_tmp_csv_file,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_latest_measurement_config() -> list[int]:
    """return all measurement ids from latest traceroute measurement config"""
    timestamp = None
    latest_config = None

    for file in path_settings.MEASUREMENTS_CONFIG.iterdir():
        if "traceroute" in file.name:
            logger.debug(f"found traceroute config:: {file.name}")

            new_config = load_json(file)
            new_timestamp = new_config["start_time"].split(".")[0]
            new_timestamp = datetime.strptime(new_timestamp, "%Y-%m-%d %H:%M:%S")

            if not timestamp:
                timestamp = new_timestamp
                latest_config = new_config

            if new_timestamp >= timestamp:
                latest_config = new_config
                timestamp = new_timestamp

    if latest_config:
        return latest_config


def get_all_measurement_ids() -> list[int]:
    """retrive all traceroute measurement ids from config file"""
    ids = set()
    for file in path_settings.MEASUREMENTS_CONFIG.iterdir():
        if "traceroute" in file.name:
            logger.info(f"Found traceroute measurement config:: {file.name}")
            config = load_json(file)
            ids.update(config["ids"])

    return ids


def get_vp_distance_matrix(vp_coordinates: dict, target_addr_list: list[str]) -> dict:
    vp_distance_matrix = defaultdict(dict)
    vp_coordinates = sorted(vp_coordinates.items(), key=lambda x: x[0])

    logger.info(
        f"Calculating VP distance matrix for {len(vp_coordinates)}x{len(vp_coordinates)}"
    )

    for i in tqdm(range(len(vp_coordinates))):
        vp_i, vp_i_coordinates = vp_coordinates[i]

        if vp_i not in target_addr_list:
            continue

        for j in range(len(vp_coordinates)):
            vp_j, vp_j_coordinates = vp_coordinates[j]
            if vp_i == vp_j:
                continue

            distance = haversine(vp_i_coordinates, vp_j_coordinates)
            vp_distance_matrix[vp_i][vp_j] = distance
            vp_distance_matrix[vp_j][vp_i] = distance

    return vp_distance_matrix


def filter_default_geoloc(vps: list, min_dist_to_default: int = 10) -> dict:
    """filter vps with coordinates too close from their country's default location"""
    countries = load_countries_info()

    valid_vps = []
    for vp in vps:
        try:
            country_geo = countries[vp["country_code"]]
        except KeyError:
            logger.warning(f"error country code {vp['country_code']} is unknown")
            continue

        dist = distance(
            country_geo["default_lat"],
            vp["lat"],
            country_geo["default_lon"],
            vp["lon"],
        )

        # Keep VPs that are away from default country geolocation
        if dist > min_dist_to_default:
            valid_vps.append(vp)
        else:
            logger.info(
                f"{vp['address_v4']}/{vp['id']}::Probed removed because of default geolocation"
            )

    return valid_vps


async def meshed_pings_schedule(update_meshed_pings: bool = True) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    measurement_schedule = []
    cached_meshed_ping_vps = None
    vps = load_vps(clickhouse_settings.VPS_RAW_TABLE)
    targets = load_targets(clickhouse_settings.VPS_RAW_TABLE)

    logger.info(
        f"Ping Schedule for:: {len(targets)} targets, {len(vps)} VPs per target"
    )

    try:
        cached_meshed_ping_vps = get_vps_ids(clickhouse_settings.VPS_MESHED_PINGS_TABLE)
        logger.info(
            f"VPs meshed ping table:: {clickhouse_settings.VPS_MESHED_PINGS_TABLE} already exists"
        )
        logger.info(f"Data will be updated instead completely renewed")
    except ClickHouseException as e:
        logger.warning(e)
        pass

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    for target in targets:
        # filter vps based on their ID and if they already pinged the target
        if cached_meshed_ping_vps and update_meshed_pings:
            try:
                cached_vp_ids = cached_meshed_ping_vps[target["addr"]]
            except KeyError:
                cached_vp_ids = []

            vp_ids = [vp["id"] for vp in vps]
            filtered_vp_ids = list(set(vp_ids).symmetric_difference(set(cached_vp_ids)))
        else:
            filtered_vp_ids = [vp["id"] for vp in vps]

        for i in range(0, len(filtered_vp_ids), batch_size):
            batch_vps = filtered_vp_ids[i : (i + batch_size)]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target["addr"],
                    [vp_id for vp_id in batch_vps if vp_id != target["id"]],
                )
            )

    logger.info(f"Total number of measurement schedule:: {len(measurement_schedule)}")

    return measurement_schedule


def meshed_traceroutes_schedule(max_nb_traceroutes: int = 100) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    vps = load_vps(clickhouse_settings.VPS_RAW_TABLE)
    vps_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

    # get vp id per addr
    vp_addr_to_id = {}
    for vp in vps:
        vp_addr_to_id[vp["addr"]] = vp["id"]

    # parse distance matrix
    ordered_distance_matrix = {}
    for vp in vps:
        distances = vps_distance_matrix[vp["addr"]]
        ordered_distance_matrix[vp["addr"]] = sorted(
            distances.items(), key=lambda x: x[-1]
        )

    traceroute_targets_per_vp = defaultdict(list)
    for vp in vps:
        closest_vp_addrs = ordered_distance_matrix[vp["addr"]]
        # only take targets outside of VPs subnet for mesehd traceroutes
        i = 0
        for addr, _ in closest_vp_addrs:
            if get_prefix_from_ip(addr) != vp["subnet"]:
                traceroute_targets_per_vp[vp["id"]].append(addr)
                i += 1

            if i > max_nb_traceroutes:
                break

    # group VPs that must trace the same target to maximize parallel measurements
    traceroute_schedule = defaultdict(set)
    for vp_id, closest_vp_addrs in traceroute_targets_per_vp.items():
        for addr in closest_vp_addrs:
            traceroute_schedule[addr].add(vp_id)

    logger.info(
        f"Traceroute Schedule for:: {len(traceroute_schedule.values())} targets"
    )

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    measurement_schedule = []
    for target, vps in traceroute_schedule.items():
        vps = list(vps)
        for i in range(0, len(vps), batch_size):
            batch_vps = vps[i : (i + batch_size)]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target,
                    [id for id in batch_vps],
                )
            )

    return measurement_schedule


def country_filtering(vps: list, countries: dict) -> list:
    filtered_vps = []
    for vp in vps:

        try:
            country_geo = countries[vp["country_code"]]
        except KeyError:
            print(f"error country code {vp['country_code']} is unknown")
            continue

        country_lat = float(country_geo["default_lat"])
        country_lon = float(country_geo["default_lon"])

        dist = distance(country_lat, vp["lat"], country_lon, vp["lon"])

        if dist < 5:
            filtered_vps.append(vp["addr"])

    logger.info(f"{len(filtered_vps)} VPs removed due to default geoloc")

    return filtered_vps


def filter_default_country_geolocation(vps: list[dict]) -> None:
    """filter VPs if there geolocation correspond to their country's default geoloc"""
    if not (path_settings.DATASET / "country_filtered_vps.json").exists():
        countries = load_countries_info()
        country_filtered_vps = country_filtering(vps, countries)
        dump_json(
            country_filtered_vps, path_settings.DATASET / "country_filtered_vps.json"
        )


def filter_wrongful_geolocation(
    targets: list[dict], vps: list[dict], meshed_pings_table: str
) -> None:
    """remove VPs based on SOI violation condition"""
    target_addr_list = [target["addr"] for target in targets]

    vp_coordinates = {}
    for vp in vps:
        vp_coordinates[vp["addr"]] = vp["lat"], vp["lon"]

    if not path_settings.VPS_PAIRWISE_DISTANCE.exists():
        vp_distance_matrix = get_vp_distance_matrix(vp_coordinates, target_addr_list)
        dump_json(vp_distance_matrix, path_settings.VPS_PAIRWISE_DISTANCE)

    if not (path_settings.DATASET / "wrongly_geolocated_probes.json").exists():
        vp_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

        rtt_per_src_dst = get_pings_per_src_dst(meshed_pings_table, threshold=300)

        removed_probes = compute_remove_wrongly_geolocated_probes(
            rtt_per_src_dst,
            vp_coordinates,
            vp_distance_matrix,
        )

        dump_json(
            removed_probes, path_settings.DATASET / "wrongly_geolocated_probes.json"
        )

        logger.info(f"Removing {len(removed_probes)} probes")


async def filter_vps() -> None:
    """filter VPs based on default geoloc and SOI violatio"""
    targets = load_targets(clickhouse_settings.VPS_RAW_TABLE)
    vps = load_vps(clickhouse_settings.VPS_RAW_TABLE)
    vps.extend(targets)

    filter_default_country_geolocation(vps)
    filter_wrongful_geolocation(
        targets, vps, clickhouse_settings.VPS_MESHED_PINGS_TABLE
    )

    if not (path_settings.DATASET / "removed_vps.json").exists():
        wrongfuly_geolocated_vps = load_json(
            path_settings.DATASET / "wrongly_geolocated_probes.json"
        )
        default_geoloc_vps = load_json(
            path_settings.DATASET / "country_filtered_vps.json"
        )

        removed_vps = list(set(wrongfuly_geolocated_vps).union(default_geoloc_vps))

        dump_json(removed_vps, path_settings.DATASET / "removed_vps.json")

    removed_vps = load_json(path_settings.DATASET / "removed_vps.json")
    logger.info(f"{len(removed_vps)} VPs removed")

    vps = load_vps(clickhouse_settings.VPS_RAW_TABLE)

    # create filtered table
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreateVPsTable().aio_execute(
            client=client,
            table_name=clickhouse_settings.VPS_FILTERED_TABLE,
        )

    csv_data = []
    for vp in vps:
        if vp["addr"] in removed_vps:
            continue

        row = []
        for val in vp.values():
            row.append(f"{val}")

        row = ",".join(row)
        csv_data.append(row)

    tmp_file_path = create_tmp_csv_file(csv_data)

    await InsertFromCSV().execute(
        table_name=clickhouse_settings.VPS_FILTERED_TABLE, in_file=tmp_file_path
    )

    tmp_file_path.unlink()


def get_updated_vps(latest_vps: list[dict], prev_vps: list[dict]) -> list[dict]:
    """return vps that were prevly unknown"""
    updated_vps = []

    prev_vps_per_id = {}
    for vp in prev_vps:
        prev_vps_per_id[vp["id"]] = vp

    for vp in latest_vps:
        prev_vp = prev_vps_per_id[vp["id"]]
        # check if value are persistent
        if (
            vp["id"] not in prev_vps_per_id
            or vp["asn_v4"] != prev_vp["asn_v4"]
            or vp["country_code"] != prev_vp["country_code"]
            or vp["lat"] != prev_vp["lat"]
            or vp["lon"] != prev_vp["lon"]
        ):
            updated_vps.append(vp)

    return updated_vps


async def vps_init(
    update_meshed_pings: bool = True, update_meshed_traceroutes: bool = True
) -> None:
    # 1. dowload all VPs
    # api = RIPEAtlasAPI()
    # vps = await api.get_vps()
    # await api.insert_vps(vps, clickhouse_settings.VPS_RAW_TABLE)

    # 2. perform meshed pings
    # pings_schedule = await meshed_pings_schedule(update_meshed_pings)
    # await RIPEAtlasProber(
    #     probing_type="ping",
    #     probing_tag="meshed_pings",
    #     output_table=clickhouse_settings.VPS_MESHED_PINGS_TABLE,
    # ).main(pings_schedule)

    # # 3. filter VPs with default geoloc and SOI violoation
    # await filter_vps()

    # # 4. perform meshed traceroutes
    traceroutes_schedule = meshed_traceroutes_schedule()
    await RIPEAtlasProber(
        probing_type="traceroute",
        probing_tag="meshed_traceroutes",
        output_table=clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE,
    ).main(traceroutes_schedule)


# debugging, testing
if __name__ == "__main__":
    traceroutes_schedule = meshed_traceroutes_schedule()
