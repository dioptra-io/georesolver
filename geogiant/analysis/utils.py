import asyncio
from tqdm import tqdm
from collections import defaultdict
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import GetVPs
from geogiant.common.geoloc import haversine
from geogiant.common.files_utils import dump_json
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def load_vps() -> list:
    """retrieve all VPs from clickhouse"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = await GetVPs().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )

    return vps


async def compute_pairwise_distance() -> dict:
    """calculate the matrix distance between all VPs"""
    vp_distance_matrix = defaultdict(list)
    vps: list = await load_vps()

    vps_coordinate = {}
    for vp in vps:
        vps_coordinate[vp["addr"]] = (vp["lat"], vp["lon"])

    for vp_i, vp_i_coordinates in tqdm(vps_coordinate.items()):
        for vp_j, vp_j_coordinates in vps_coordinate.items():

            if vp_i == vp_j:
                continue

            distance = haversine(vp_i_coordinates, vp_j_coordinates)
            vp_distance_matrix[vp_i].append((vp_j, distance))

        # take only the first 1_000 closest VPs
        distances = sorted(vp_distance_matrix[vp_i], key=lambda x: x[1])
        vp_distance_matrix[vp_i] = distances[:1_000]

    dump_json(vp_distance_matrix, path_settings.VPS_PAIRWISE_DISTANCE)


if __name__ == "__main__":
    asyncio.run(compute_pairwise_distance())
