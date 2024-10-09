from pyasn import pyasn
from random import sample
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from numpy import mean, median
from collections import defaultdict
from pych_client import ClickHouseClient

from geogiant.clickhouse import Query
from geogiant.common.geoloc import distance
from geogiant.common.utils import (
    TargetScores,
    get_parsed_vps,
    parse_target,
    get_ecs_vps,
    filter_vps_last_mile_delay,
    get_no_ping_vp,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.files_utils import load_pickle, load_json
from geogiant.common.queries import (
    load_targets,
    load_vps,
    get_min_rtt_per_vp,
)
from geogiant.internet_scale.internet_scale_pings import get_geo_resolver_schedule
from geogiant.no_ping_geoloc.extended_scores import get_scores
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

PING_TABLE = "pings_internet_scale"
ECS_TABLE = "internet_scale_mapping_ecs"
VPS_ECS_MAPPING_TABLE = "vps_mapping_ecs"
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnets_filtered.json"
SCORE_PATH = path_settings.RESULTS_PATH / "extended_evaluation/score.pickle"


def load_hostnames() -> dict:
    selected_hostnames_per_cdn_per_ns = load_json(
        path_settings.DATASET
        / f"hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json"
    )

    selected_hostnames = set()
    selected_hostnames_per_cdn = defaultdict(list)
    for ns in selected_hostnames_per_cdn_per_ns:
        for org, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
            selected_hostnames.update(hostnames)
            selected_hostnames_per_cdn[org].extend(hostnames)

    logger.info(f"{len(selected_hostnames)=}")

    return selected_hostnames_per_cdn, selected_hostnames


class GetClosestAddr(Query):
    def statement(self, table_name: str, **kwargs):
        try:
            vp_filter = kwargs["vp_filter"]
        except KeyError:
            raise RuntimeError(f"vp_filter parameter necessary for {self.__class__}")

        vp_filter_statement = "".join([f",toIPv4('{s}')" for s in vp_filter])[1:]
        vp_filter_statement = f"AND src_addr IN ({vp_filter_statement})"

        return f"""
        SELECT
            src_prefix,
            groupUniqArray(dst_addr) as landmarks
        FROM {self.settings.CLICKHOUSE_DATABASE}.{table_name}
        WHERE 
            min > -1
            AND min <= 2
            {vp_filter_statement}            
        GROUP BY src_prefix
        """


def get_landmarks_addr(table_name: str, vp_filter: list[str]) -> dict:
    """get all IP addresses close to a list of VPs"""
    landmarks_per_vp = {}

    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetClosestAddr().execute(
            client=client, table_name=table_name, vp_filter=vp_filter
        )

    for row in rows:
        landmarks_per_vp[row["src_prefix"]] = row["landmarks"]

    return landmarks_per_vp


def get_extended_scores(score_schedule: list[dict], output_path: Path) -> dict:

    _, selected_hostnames = load_hostnames()

    scores = get_scores(
        hostnames=selected_hostnames,
        score_schedule=score_schedule,
        target_ecs_table="vps_mapping_ecs",
        vps_ecs_table="vps_mapping_ecs",
        landmarks_ecs_table="internet_scale_mapping_ecs",
        output_path=output_path,
    )

    return scores


def load_targets_from_score(
    subnet_scores: dict[list], addr_per_subnet: dict
) -> list[str]:
    targets = []
    for target_subnet in subnet_scores:
        target_addr = addr_per_subnet[target_subnet][0]
        targets.append(target_addr)

    return targets


def load_geo_resolver_vps(targets: list[str]) -> dict:
    target_scores: TargetScores = load_pickle(
        path_settings.RESULTS_PATH
        / "tier4_evaluation/scores__best_hostname_geo_score_20_BGP_3_hostnames_per_org_ns.pickle"
    )
    target_scores = target_scores.score_answer_subnets

    asndb = pyasn(str(path_settings.RIB_TABLE))
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb, removed_vps)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )

    logger.info(f"Targets in schedule:: {len(targets)}")

    target_geo_resolver_vps = get_geo_resolver_schedule(
        targets=[target["addr"] for target in targets],
        subnet_scores=target_scores,
        vps_per_subnet=vps_per_subnet,
        last_mile_delay=last_mile_delay,
        vps_coordinates=vps_coordinates,
    )

    return target_geo_resolver_vps


def generate_extended_score_file(
    vps: list[str],
    geo_resolver_vps: dict,
    landmarks_per_vp: dict,
) -> None:
    score_schedule = []
    for target, vps in geo_resolver_vps.items():
        target_subnet = get_prefix_from_ip(target)
        vps_subnet = set()
        landmarks_subnet = set()

        # select all landmarks on which to calculate score
        for vp_addr in vps:
            # add vp subnet for score calculation
            vp_subnet = get_prefix_from_ip(vp_addr)
            vps_subnet.add(vp_subnet)

            # add all associated vps landmarks
            try:
                vp_landmarks = landmarks_per_vp[vp_subnet]

                if len(vp_landmarks) > 100:
                    vp_landmarks = sample(vp_landmarks, 100)

                vp_landmarks = [
                    get_prefix_from_ip(landmark) for landmark in vp_landmarks
                ]

            except KeyError:
                logger.warning(f"{vp_addr} not have any landmark")
                continue

            landmarks_subnet.update(vp_landmarks)

        logger.info(f"{target_subnet=}:: {len(landmarks_subnet)} subnet landmarks")

        score_schedule.append(
            (
                {
                    "target_subnet": target_subnet,
                    "vps_subnet": vps_subnet,
                    "landmarks_subnet": landmarks_subnet,
                }
            )
        )

    get_extended_scores(score_schedule=score_schedule, output_path=SCORE_PATH)


def get_median_error(
    targets: list[dict],
    scores: dict,
    vps_per_subnet: dict,
    vps_coordinates: dict,
    last_mile_delay: dict,
) -> None:
    results = {}
    for target in tqdm(targets):
        target = parse_target(target, asndb)

        try:

            target_score = scores[target["subnet"]]
        except KeyError as e:
            raise RuntimeError(f"no score for:: {target['subnet']=}, {e}")

        # get vps, function of their subnet ecs score
        ecs_vps = get_ecs_vps(
            target["subnet"], target_score, vps_per_subnet, last_mile_delay, 10_000
        )

        # remove vps that have a high last mile delay
        ecs_vps = filter_vps_last_mile_delay(ecs_vps, last_mile_delay, 2)

        no_ping_vp = get_no_ping_vp(
            target,
            target_score,
            vps_per_subnet,
            vps_coordinates,
        )

        results[target["addr"]] = {
            "target": target,
            "no_ping_vp": no_ping_vp,
        }

    d_errors = []
    for target, target_result in results.items():
        d_error = target_result["no_ping_vp"]["d_error"]
        d_errors.append(d_error)

    median_error = round(median(d_errors), 2)

    logger.info(
        f"Extended ping median error:: {median_error} [km], {len(geo_resoler_vps)} targets"
    )


if __name__ == "__main__":
    calculate_scores = False
    evaluation = True

    # load targets / vps / pings / landmarks
    asndb = pyasn(str(path_settings.RIB_TABLE))
    targets = load_targets(clickhouse_settings.VPS_FILTERED_TABLE)
    # targets = targets[:1]
    target_coordinates = {}
    for target in targets:
        target_coordinates[target["addr"]] = (target["lat"], target["lon"])
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    vp_filter = [vp["addr"] for vp in vps]
    landmarks_per_vp = get_landmarks_addr(PING_TABLE, vp_filter)
    geo_resoler_vps = load_geo_resolver_vps(targets)
    vps_per_subnet, vps_coordinates = get_parsed_vps(vps, asndb)
    last_mile_delay = get_min_rtt_per_vp(
        clickhouse_settings.VPS_MESHED_TRACEROUTE_TABLE
    )

    if calculate_scores:
        generate_extended_score_file(
            vps=vps,
            geo_resolver_vps=geo_resoler_vps,
            landmarks_per_vp=landmarks_per_vp,
        )

    if evaluation:
        target_scores = load_pickle(SCORE_PATH)

        extended_target_scores = defaultdict(list)
        for target_subnet, scores in target_scores.items():
            for vp_subnet, vp_score in scores["vps_score"]:

                try:
                    landmarks = landmarks_per_vp[vp_subnet]
                    landmarks = landmarks[:20]
                except KeyError:
                    extended_target_scores[target_subnet].append((vp_subnet, vp_score))
                    continue

                landmarks_subnet = [
                    get_prefix_from_ip(landmark) for landmark in landmarks
                ]

                landmark_scores = target_scores[target_subnet]["landmarks_score"]

                extended_score = []
                for subnet, score in landmark_scores:
                    if subnet in landmarks_subnet:
                        extended_score.append(score)

                # extended vp score = avg (vp score + landmark scores)
                extended_vp_score = vp_score + mean(extended_score)
                # extended_vp_score = mean(extended_score)
                extended_target_scores[target_subnet].append(
                    (vp_subnet, extended_vp_score)
                )

            extended_target_scores[target_subnet] = sorted(
                extended_target_scores[target_subnet],
                key=lambda x: x[1],
                reverse=True,
            )

        vps_score = {}
        for target_subnet, scores in target_scores.items():
            vps_score[target_subnet] = [vp_score for vp_score in scores["vps_score"]]

        get_median_error(
            targets=targets,
            scores=vps_score,
            vps_per_subnet=vps_per_subnet,
            vps_coordinates=vps_coordinates,
            last_mile_delay=last_mile_delay,
        )

        get_median_error(
            targets=targets,
            scores=extended_target_scores,
            vps_per_subnet=vps_per_subnet,
            vps_coordinates=vps_coordinates,
            last_mile_delay=last_mile_delay,
        )


# for target_addr, selected_vps in tqdm(geo_resoler_vps.items()):
#     target_subnet = get_prefix_from_ip(target_addr)

#     # calculate extended score
#     # (vp score and landmark associated scores)
#     target_extended_scores = []
#     target_scores = []
#     for vp_addr in selected_vps:
#         vp_subnet = get_prefix_from_ip(vp_addr)
#         target_vp_scores = target_score[target_subnet]
#         try:
#             vp_landmarks = landmarks_per_vp[vp_addr]
#         except KeyError:
#             vp_landmarks = []

#         for subnet, score in target_vp_scores:
#             if subnet == vp_subnet or subnet in vp_landmarks:
#                 target_extended_scores.append(score)

#             if subnet == vp_subnet:
#                 target_scores.append(score)

#         extended_scores[target_addr].append(
#             (vp_addr, mean(target_extended_scores))
#         )
#         scores[target_addr].append((vp_addr, target_scores))

#     # get distance of vp with best score
#     extended_scores[target_addr] = sorted(
#         extended_scores[target_addr], key=lambda x: x[-1], reverse=True
#     )

#     # get distance of vp with best score
#     scores[target_addr] = sorted(
#         scores[target_addr], key=lambda x: x[-1], reverse=True
#     )

# distance_errors = []
# for target, vp_scores in extended_scores.items():

#     zero_ping_vp_addr, zero_ping_vp_score = vp_scores[0]

#     vp_lat, vp_lon, _, _ = vps_coordinates[zero_ping_vp_addr]
#     target_lat, target_lon = target_coordinates[target_addr]

#     distance_errors.append(distance(target_lat, vp_lat, target_lon, vp_lon))

# median_error = round(median(distance_errors), 2)

# logger.info(
#     f"Extended ping median error:: {median_error} [km], {len(geo_resoler_vps)} targets"
# )

# distance_errors = []
# for target, vp_scores in scores.items():

#     zero_ping_vp_addr, zero_ping_vp_score = vp_scores[0]

#     vp_lat, vp_lon, _, _ = vps_coordinates[zero_ping_vp_addr]
#     target_lat, target_lon = target_coordinates[target_addr]

#     distance_errors.append(distance(target_lat, vp_lat, target_lon, vp_lon))

# median_error = round(median(distance_errors), 2)

# logger.info(
#     f"ping median error:: {median_error} [km], {len(geo_resoler_vps)} targets"
# )
