import asyncio

from collections import defaultdict
from loguru import logger
from tqdm import tqdm
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import CreateVPsTable, InsertFromCSV
from geogiant.common.files_utils import (
    load_json,
    dump_json,
    load_countries_info,
    create_tmp_csv_file,
)
from geogiant.common.queries import (
    load_targets,
    load_vps,
    get_pings_per_src_dst,
    load_all_vps,
)
from geogiant.common.geoloc import distance, haversine
from geogiant.common.settings import PathSettings, ClickhouseSettings, ConstantSettings

constant_settings = ConstantSettings()
path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


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


def compute_remove_wrongly_geolocated_probes(
    rtts_per_srcs_dst: dict,
    vp_coordinates: dict,
    vp_distance_matrix: dict[dict],
) -> set:
    speed_of_internet_violations_per_ip = defaultdict(set)

    for dst, rtts_per_src in rtts_per_srcs_dst.items():
        if dst not in vp_coordinates:
            continue

        if dst not in vp_distance_matrix:
            continue

        for probe, min_rtt in rtts_per_src.items():
            if probe not in vp_distance_matrix[dst]:
                continue
            max_theoretical_distance = (
                constant_settings.SPEED_OF_INTERNET * min_rtt / 1000
            ) / 2
            if vp_distance_matrix[dst][probe] > max_theoretical_distance:
                # Impossible distance
                speed_of_internet_violations_per_ip[dst].add(probe)
                speed_of_internet_violations_per_ip[probe].add(dst)

    # Greedily remove the IP address with the more SOI violations
    n_violations = sum([len(x) for x in speed_of_internet_violations_per_ip.values()])
    removed_probes = set()
    while n_violations > 0:
        logger.info("Violations:", n_violations)
        # Remove the IP address with the highest number of SOI violations
        worse_ip, speed_of_internet_violations = max(
            speed_of_internet_violations_per_ip.items(), key=lambda x: len(x[1])
        )
        for (
            ip,
            speed_of_internet_violations,
        ) in speed_of_internet_violations_per_ip.items():
            speed_of_internet_violations.discard(worse_ip)
        del speed_of_internet_violations_per_ip[worse_ip]
        removed_probes.add(worse_ip)
        n_violations = sum(
            [len(x) for x in speed_of_internet_violations_per_ip.values()]
        )

    return list(removed_probes)


async def main() -> None:
    """filter VPs based on default geoloc and SOI violatio"""
    # here load VPs and country file info
    targets = load_targets(clickhouse_settings.VPS_RAW)
    vps = load_vps(clickhouse_settings.VPS_RAW)

    target_addr_list = [target["addr"] for target in targets]

    if not (path_settings.DATASET / "country_filtered_vps.json").exists():
        countries = load_countries_info()
        country_filtered_vps = country_filtering(vps, countries)
        dump_json(
            country_filtered_vps, path_settings.DATASET / "country_filtered_vps.json"
        )

    vp_coordinates = {}
    for vp in vps:
        vp_coordinates[vp["addr"]] = vp["lat"], vp["lon"]

    if not path_settings.VPS_PAIRWISE_DISTANCE.exists():
        vp_distance_matrix = get_vp_distance_matrix(vp_coordinates, target_addr_list)
        dump_json(vp_distance_matrix, path_settings.VPS_PAIRWISE_DISTANCE)

    if not (path_settings.DATASET / "wrongly_geolocated_probes.json").exists():
        vp_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

        rtt_per_src_dst = get_pings_per_src_dst(
            clickhouse_settings.PING_VPS_TO_TARGET, threshold=300
        )

        removed_probes = compute_remove_wrongly_geolocated_probes(
            rtt_per_src_dst,
            vp_coordinates,
            vp_distance_matrix,
        )

        dump_json(
            removed_probes, path_settings.DATASET / "wrongly_geolocated_probes.json"
        )

        logger.info(f"Removing {len(removed_probes)} probes")

    # finnaly remove probes that were filtered and create new table
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

    vps = load_all_vps(clickhouse_settings.VPS_RAW)

    # create filtered table
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreateVPsTable().aio_execute(
            client=client,
            table_name="filtered_vps",
        )

    csv_data = []
    for vp in vps:
        if vp["address_v4"] in removed_vps:
            continue

        row = []
        for val in vp.values():
            row.append(f"{val}")

        row = ",".join(row)
        csv_data.append(row)

    tmp_file_path = create_tmp_csv_file(csv_data)

    await InsertFromCSV().execute(table_name="filtered_vps", in_file=tmp_file_path)

    tmp_file_path.unlink()


if __name__ == "__main__":
    asyncio.run(main())
