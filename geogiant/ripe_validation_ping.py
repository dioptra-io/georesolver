import asyncio
import time

from tqdm import tqdm
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import GetVPs, CreatePingTable, InsertFromCSV
from geogiant.prober import RIPEAtlasAPI
from geogiant.common.files_utils import create_tmp_csv_file, load_json
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


async def get_measurement_schedule(dry_run: bool = False) -> dict:
    """for each target and subnet target get measurement vps

    Returns:
        dict: vps per target to make a measurement
    """
    vps = await load_vps()
    targets = await load_targets()

    logger.info(
        f"Ping Schedule for:: {len(targets)} targets, {len(vps)} VPs per target"
    )

    if dry_run:
        targets = targets[10:13]
        vps = vps[20:25]
        logger.debug(f"Dry run:: {len(targets)}, {len(vps)} VPs per target")

    batch_size = RIPEAtlasAPI().settings.MAX_VP
    measurement_schedule = []
    for target in targets:
        for i in range(0, len(vps), 1):
            batch_vps = vps[i * batch_size : (i + 1) * batch_size]

            if not batch_vps:
                break

            measurement_schedule.append(
                (
                    target["addr"],
                    [vp["id"] for vp in batch_vps if vp["id"] != target["id"]],
                )
            )

    return measurement_schedule


async def retrieve_pings(ids: list[int]) -> list[dict]:
    """retrieve all ping measurements from a list of measurement ids"""
    csv_data = []
    for id in tqdm(ids):
        ping_results = await RIPEAtlasAPI().get_ping_results(id)
        csv_data.extend(ping_results)

        time.sleep(0.1)

    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreatePingTable().aio_execute(
            client=client, table_name="ping_vps_to_targets"
        )
        await InsertFromCSV().execute(
            table_name="ping_vps_to_targets",
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def main() -> None:
    # measurement_schedule = await get_measurement_schedule(dry_run=False)
    # await RIPEAtlasProber("ping").main(measurement_schedule)

    # retrive ping measurements and insert them into clickhouse db
    config_file = "ping__b74a94f7-03e4-41dd-83f3-2130dad140eb.json"
    measurement_config = load_json(
        RIPEAtlasAPI().settings.MEASUREMENTS_CONFIG / config_file
    )

    await retrieve_pings(measurement_config["ids"])


if __name__ == "__main__":
    logger.info("Starting validation Ping measurement on all RIPE atlas anchors")
    asyncio.run(main())
