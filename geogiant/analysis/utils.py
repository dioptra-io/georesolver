from tqdm import tqdm
from collections import defaultdict

from geogiant.common.geoloc import haversine
from geogiant.common.files_utils import load_json, dump_json
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


def compute_pairwise_distance() -> dict:
    """calculate the matrix distance between all VPs"""
    vp_distance_matrix = defaultdict(list)
    vps: list = load_json(path_settings.OLD_VPS)

    vps_coordinate = {}
    for vp in vps:
        long, lat = vp["geometry"]["coordinates"]
        vps_coordinate[vp["address_v4"]] = lat, long

    for vp_i, vp_i_coordinates in tqdm(vps_coordinate.items()):
        for vp_j, vp_j_coordinates in vps_coordinate.items():

            if vp_i == vp_j:
                continue

            distance = haversine(vp_i_coordinates, vp_j_coordinates)
            vp_distance_matrix[vp_i].append((vp_j, distance))

        # take only the first 1_000 closest VPs
        distances = sorted(distances, key=lambda x: x[1])
        vp_distance_matrix[vp] = distances[:1_000]

    dump_json(vp_distance_matrix, path_settings.OLD_VPS_PAIRWISE_DISTANCE)


if __name__ == "__main__":
    compute_pairwise_distance()
