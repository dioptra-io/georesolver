"""VPs initialization functions"""

import asyncio

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from pych_client import AsyncClickHouseClient
from pych_client.exceptions import ClickHouseException

from georesolver.agent.insert_process import retrieve_pings, retrieve_traceroutes
from georesolver.prober import RIPEAtlasAPI, RIPEAtlasProber
from georesolver.clickhouse import CreateVPsTable, Query
from georesolver.clickhouse.queries import (
    load_vps,
    load_targets,
    get_vps_ids_per_target,
    get_pings_per_src_dst,
    get_min_rtt_per_vp,
    change_table_name,
)
from georesolver.common.geoloc import (
    distance,
    haversine,
    compute_remove_wrongly_geolocated_probes,
)
from georesolver.common.files_utils import (
    load_countries_info,
    dump_json,
    load_json,
    create_tmp_csv_file,
)
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
ch_settings = ClickhouseSettings()


def table_name_with_date(table_name: str) -> str:
    """return a new table name with date appended to it"""
    d = datetime.now()
    month = "0" + str(d.month) if d.month < 10 else str(d.month)
    day = "0" + str(d.day) if d.day < 10 else str(d.day)
    return table_name + f"_{d.year}_{month}_{day}"


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


def get_vp_distance_matrix(vp_coordinates: dict, target_ids: list[str]) -> dict:
    """compute pairwise distance between each VPs and a list of target ids (RIPE Atlas anchors)"""
    logger.info(
        f"Calculating VP distance matrix for {len(vp_coordinates)}x{len(target_ids)}"
    )

    vp_distance_matrix = defaultdict(dict)
    # for better efficiency only compute pairwise distance for the targets
    for vp_i_id in tqdm(target_ids):
        vp_i_coordinates = vp_coordinates[vp_i_id]

        for vp_j_id, vp_j_coordinates in vp_coordinates.items():
            if vp_i_id == vp_j_id:
                continue

            distance = haversine(vp_i_coordinates, vp_j_coordinates)

            vp_distance_matrix[vp_i_id][vp_j_id] = distance
            vp_distance_matrix[vp_j_id][vp_i_id] = distance

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
    vps = load_vps(ch_settings.VPS_RAW_TABLE)
    targets = load_targets(ch_settings.VPS_RAW_TABLE)

    logger.info(
        f"Ping Schedule for:: {len(targets)} targets, {len(vps)} VPs per target"
    )

    try:
        cached_meshed_ping_vps = get_vps_ids_per_target(
            ch_settings.VPS_MESHED_PINGS_TABLE
        )
        logger.info(
            f"VPs meshed ping table:: {ch_settings.VPS_MESHED_PINGS_TABLE} already exists"
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
    vps = load_vps(ch_settings.VPS_RAW_TABLE)
    vps_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

    # get vp id per addr
    vp_per_id = {}
    vp_per_addr = {}
    for vp in vps:
        vp_per_id[str(vp["id"])] = vp
        vp_per_addr[vp["addr"]] = vp

    # parse distance matrix
    ordered_distance_matrix = {}
    for vp in vps:
        distances = vps_distance_matrix[str(vp["id"])]
        ordered_distance_matrix[str(vp["id"])] = sorted(
            distances.items(), key=lambda x: x[-1]
        )

    traceroute_targets_per_vp = defaultdict(list)
    for vp in vps:
        closest_vp_ids = ordered_distance_matrix[str(vp["id"])]
        closest_vp_addrs = [vp_per_id[id]["addr"] for id, _ in closest_vp_ids]
        # only take targets outside of VPs subnet for mesehd traceroutes
        i = 0
        for addr in closest_vp_addrs:
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


def filter_default_country_geolocation(vps: list[dict]) -> None:
    """filter VPs if there geolocation correspond to their country's default geoloc"""
    filtered_vps = []
    if not (path_settings.DATASET / "country_filtered_vps.json").exists():
        countries = load_countries_info()
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
                filtered_vps.append((vp["id"]), vp["addr"])

        logger.info(f"{len(filtered_vps)} VPs removed due to default geoloc")

        dump_json(filtered_vps, path_settings.DATASET / "country_filtered_vps.json")

    return filtered_vps


def filter_wrongful_geolocation(
    targets: list[dict], vps: list[dict], meshed_pings_table: str
) -> None:
    """remove VPs based on SOI violation condition"""
    target_ids = [target["id"] for target in targets]

    vp_per_id = {}
    vp_per_addr = {}
    vp_coordinates = {}
    for vp in vps:
        vp_per_id[str(vp["id"])] = vp
        vp_per_addr[vp["addr"]] = vp
        vp_coordinates[str(vp["id"])] = vp["lat"], vp["lon"]

    if not path_settings.VPS_PAIRWISE_DISTANCE.exists():
        vp_distance_matrix = get_vp_distance_matrix(vp_coordinates, target_ids)
        dump_json(vp_distance_matrix, path_settings.VPS_PAIRWISE_DISTANCE)

    if (path_settings.DATASET / "wrongly_geolocated_probes.json").exists():
        vp_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

        ping_per_target_per_vp = get_pings_per_src_dst(
            meshed_pings_table, threshold=300
        )

        removed_probes = compute_remove_wrongly_geolocated_probes(
            ping_per_target_per_vp=ping_per_target_per_vp,
            vp_coordinates=vp_coordinates,
            vp_distance_matrix=vp_distance_matrix,
            vp_per_addr=vp_per_addr,
            vp_per_id=vp_per_id,
        )

        dump_json(
            removed_probes, path_settings.DATASET / "wrongly_geolocated_probes.json"
        )

        logger.info(f"Removing {len(removed_probes)} probes")

    return removed_probes


async def filter_vps() -> None:
    """filter VPs based on default geoloc and SOI violatio"""
    targets = load_targets(ch_settings.VPS_RAW_TABLE)
    vps = load_vps(ch_settings.VPS_RAW_TABLE)

    ping_filtered = filter_wrongful_geolocation(
        targets, vps, ch_settings.VPS_MESHED_PINGS_TABLE
    )
    country_filtered = filter_default_country_geolocation(vps)

    removed_vps = list(set(ping_filtered).union(set(country_filtered)))

    dump_json(removed_vps, path_settings.DATASET / "removed_vps.json")

    logger.info(f"{len(removed_vps)} VPs removed")

    return removed_vps


async def update_vps_filtered_table(
    suffixe_table: str, removed_vps: list[dict]
) -> None:
    """filter vps table, update old table if already exists"""

    # archive old table
    change_table_name(
        ch_settings.VPS_FILTERED_TABLE,
        ch_settings.VPS_FILTERED_TABLE + "_" + suffixe_table,
    )

    vps = load_vps(ch_settings.VPS_RAW_TABLE)

    filtered_ids = [id for id, _ in removed_vps]

    csv_data = []
    for vp in vps:
        if str(vp["id"]) in filtered_ids:
            continue

        row = []
        for val in vp.values():
            row.append(f"{val}")

        row = ",".join(row)
        csv_data.append(row)

    # create filtered table
    async with AsyncClickHouseClient(**ch_settings.clickhouse) as client:
        tmp_file_path = create_tmp_csv_file(csv_data)

        await CreateVPsTable().aio_execute(
            client=client,
            table_name=ch_settings.VPS_FILTERED_TABLE,
        )

        Query().execute_insert(
            client=client,
            table_name=ch_settings.VPS_FILTERED_TABLE,
            in_file=tmp_file_path,
        )

        tmp_file_path.unlink()


async def filter_low_connectivity_vps(delay_threshold: int = 2) -> None:
    """filter all VPs for which the measured minimum last mile delay above 2ms"""
    targets = load_targets(ch_settings.VPS_RAW_TABLE)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    last_mile_delay = get_min_rtt_per_vp(ch_settings.VPS_MESHED_TRACEROUTE_TABLE)
    vps.extend(targets)

    logger.info(f"Original number of VPs:: {len(vps)}")
    logger.info(f"NB vps traceroutes:: {len(last_mile_delay)}")

    # only keep vps with a last mile delay under threshold
    filtered_vps = []
    for vp in vps:
        try:
            min_rtt = last_mile_delay[vp["addr"]]
            if min_rtt <= delay_threshold:
                filtered_vps.append((vp["id"], vp["addr"]))
        except KeyError:
            continue

    logger.info(f"Remaining VPs after last mile delay filtering:: {len(filtered_vps)}")

    return filtered_vps


async def update_vps_filtered_final(
    filtered_vps: list[dict], suffixe_table: str
) -> None:
    """update vps filtered final table"""

    filtered_vps_ids = [vp_id for vp_id, _ in filtered_vps]
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)

    csv_data = []
    for vp in vps:
        if vp["id"] not in filtered_vps_ids:
            continue

        row = []
        for val in vp.values():
            row.append(f"{val}")

        row = ",".join(row)
        csv_data.append(row)

    # archive old table
    change_table_name(
        ch_settings.VPS_FILTERED_FINAL_TABLE,
        ch_settings.VPS_FILTERED_FINAL_TABLE + "_" + suffixe_table,
    )

    async with AsyncClickHouseClient(**ch_settings.clickhouse) as client:
        tmp_file_path = create_tmp_csv_file(csv_data)

        await CreateVPsTable().aio_execute(
            client=client,
            table_name=ch_settings.VPS_FILTERED_FINAL_TABLE,
        )

        Query().execute_insert(
            client=client,
            table_name=ch_settings.VPS_FILTERED_FINAL_TABLE,
            in_file=tmp_file_path,
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


async def update_vps_table(suffixe_table: str) -> None:
    """update vps table, change old vps table names with date"""
    api = RIPEAtlasAPI()
    vps = await api.get_vps()

    # archive old table
    change_table_name(
        ch_settings.VPS_RAW_TABLE, ch_settings.VPS_RAW_TABLE + "_" + suffixe_table
    )

    await api.insert_vps(vps, ch_settings.VPS_RAW_TABLE)


async def update_meshed_pings(suffixe_table: str) -> None:
    """update vps meshed pings table"""
    # archive old table
    change_table_name(
        ch_settings.VPS_MESHED_PINGS_TABLE,
        ch_settings.VPS_MESHED_PINGS_TABLE + "_" + suffixe_table,
    )

    # 2. perform meshed pings
    pings_schedule = await meshed_pings_schedule(False)
    output_path = await RIPEAtlasProber(
        probing_type="ping",
        probing_tag="meshed-pings",
        output_table=ch_settings.VPS_MESHED_PINGS_TABLE,
    ).main(pings_schedule)


async def insert_pings(config_path: Path) -> None:
    """insert pings from measurement config file"""
    batch_size: int = 1_000
    # insert pings using ids saved in config
    measurement_config = load_json(config_path)
    measurement_ids = [id for id, _ in measurement_config["ids"]]
    for i in range(0, len(measurement_ids), batch_size):
        batch_ids = measurement_ids[i : (i + batch_size)]

        logger.info(f"Batch:: {i // batch_size}/{len(measurement_ids) // batch_size}")

        await retrieve_pings(batch_ids, ch_settings.VPS_MESHED_PINGS_TABLE)


async def update_meshed_traceroutes(suffixe_table: str) -> None:
    """update vps meshed pings table"""
    # archive old table
    change_table_name(
        ch_settings.VPS_MESHED_TRACEROUTE_TABLE,
        ch_settings.VPS_MESHED_TRACEROUTE_TABLE + "_" + suffixe_table,
    )

    # 2. perform meshed pings
    traceroutes_schedule = meshed_traceroutes_schedule()
    output_path = await RIPEAtlasProber(
        probing_type="traceroute",
        probing_tag="meshed-traceroutes",
        output_table=ch_settings.VPS_MESHED_TRACEROUTE_TABLE,
    ).main(traceroutes_schedule)


async def insert_traceroutes(config_path: Path) -> None:
    """insert traceroutes from"""
    batch_size: int = 1_000
    # insert pings using ids saved in config
    measurement_config = load_json(config_path)
    measurement_ids = [id for id, _ in measurement_config["ids"]]
    for i in range(0, len(measurement_ids), batch_size):
        batch_ids = measurement_ids[i : (i + batch_size)]

        logger.info(f"Batch:: {i // batch_size}/{len(measurement_ids) // batch_size}")

        await retrieve_traceroutes(batch_ids, ch_settings.VPS_MESHED_TRACEROUTE_TABLE)


async def vps_init(old_table_suffixe: str) -> None:
    """perform all georesolver ripe atlas init related functions"""
    """perform all georesolver ripe atlas init related functions"""

    # 1. download all VPs
    await update_vps_table(old_table_suffixe)

    # 2. update vps meshed pings table
    config_path = await update_meshed_pings(old_table_suffixe)
    await insert_pings(config_path)

    # 3. filter VPs with default geoloc and SOI violation
    removed_vps = await filter_vps()
    await update_vps_filtered_table(old_table_suffixe, removed_vps)

    # # 4. perform meshed traceroutes
    config_path = await update_meshed_traceroutes(old_table_suffixe)
    await insert_traceroutes(config_path)

    # 5. create a table with only VPs with last myle delay under 2 ms
    vps_filtered = await filter_low_connectivity_vps(delay_threshold=2)
    await update_vps_filtered_final(vps_filtered, suffixe_table="None")


# debugging, testing
if __name__ == "__main__":

    old_table_suffixe = "CoNEXT_winter_submision"
    asyncio.run(vps_init(old_table_suffixe))
