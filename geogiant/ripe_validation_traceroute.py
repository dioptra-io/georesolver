import asyncio

from tqdm import tqdm
from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.common.files_utils import load_json, dump_json
from geogiant.common.geoloc import distance
from geogiant.clickhouse import GetVPs
from geogiant.prober import RIPEAtlasProber, RIPEAtlasAPI
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def load_vps() -> list:
    """retrieve all VPs from clickhouse"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = await GetVPs().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW
        )

    return vps


async def load_targets() -> list:
    """load all targets (ripe atlas anchors) from clickhouse"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        targets = await GetVPs().execute(
            client=client, table_name=clickhouse_settings.VPS_RAW, is_anchor=True
        )

    return targets


def compute_vp_distance_matrix(vps: list[dict]) -> None:
    """
    calculate distance from one VP to all the others,
    only keep the first thousands closest VPs
    """
    logger.info("Computing VP distance matrix")

    vp_pairwise_distance = defaultdict(list)
    for vp_i in tqdm(vps):
        for vp_j in vps:
            if vp_i["addr"] == vp_j["addr"]:
                continue

            d = distance(vp_i["lat"], vp_j["lat"], vp_i["lon"], vp_j["lon"])
            vp_pairwise_distance[vp_i["addr"]].append((vp_j["addr"], d))

        vp_pairwise_distance[vp_i["addr"]] = sorted(
            vp_pairwise_distance[vp_i["addr"]], key=lambda x: x[-1]
        )
        vp_pairwise_distance[vp_i["addr"]] = vp_pairwise_distance[vp_i["addr"]][:50]

    dump_json(
        data=vp_pairwise_distance, output_file=path_settings.VPS_PAIRWISE_DISTANCE
    )


async def get_measurement_schedule(dry_run: bool = False) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    vps = await load_vps()

    if dry_run:
        targets = targets[:5]
        vps = vps[:2]
        logger.debug(f"Dry run:: {len(targets)}, {len(vps)} VPs per target")

    if not path_settings.VPS_PAIRWISE_DISTANCE.exists():
        logger.debug(
            f"VP distance matrix file:: {path_settings.VPS_PAIRWISE_DISTANCE} does not exists, calculating it"
        )
        compute_vp_distance_matrix(vps)

    vps_distance_matrix = load_json(path_settings.VPS_PAIRWISE_DISTANCE)

    logger.debug(f"{len(vps_distance_matrix)=}")

    # get vp id per addr
    vp_addr_to_id = {}
    for vp in vps:
        vp_addr_to_id[vp["addr"]] = vp["id"]

    traceroute_targets_per_vp = {}
    for vp in vps:
        closest_vps = vps_distance_matrix[vp["addr"]][:50]
        closest_vps_ids = [
            (vp_addr, vp_addr_to_id[vp_addr]) for vp_addr, _ in closest_vps
        ]
        traceroute_targets_per_vp[vp["id"]] = closest_vps_ids

    logger.debug(f"{len(traceroute_targets_per_vp)=}")

    # group VPs that must trace the same target to maximize parallel measurements
    traceroute_schedule = defaultdict(set)
    for vp_id, closest_vp_ids in traceroute_targets_per_vp.items():
        for addr, id in closest_vp_ids:
            traceroute_schedule[addr].add(vp_id)

    logger.info(
        f"Traceroute Schedule for:: {len(traceroute_schedule.values())} targets"
    )
    count = 0
    for addr, ids in traceroute_schedule.items():
        count += len(ids)
    logger.info(f"Total number of traceroute:: {count}")

    batch_size = RIPEAtlasAPI().settings.MAX_MEASUREMENT
    measurement_schedule = []
    for target, vps in traceroute_schedule.items():
        vps = list(vps)
        for i in range(0, len(vps), 1):
            batch_vps = vps[i * batch_size : (i + 1) * batch_size]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target,
                    [id for id in batch_vps],
                )
            )

    return measurement_schedule


async def main() -> None:
    measurement_schedule = await get_measurement_schedule(dry_run=False)
    await RIPEAtlasProber("traceroute").main(measurement_schedule)


if __name__ == "__main__":
    logger.info("Starting validation Ping measurement on all RIPE atlas anchors")
    asyncio.run(main())
