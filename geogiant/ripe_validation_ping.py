import asyncio

from loguru import logger
from pych_client import AsyncClickHouseClient

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


async def main() -> None:
    measurement_schedule = await get_measurement_schedule(dry_run=False)
    await RIPEAtlasProber("ping").main(measurement_schedule)


if __name__ == "__main__":
    logger.info("Starting validation Ping measurement on all RIPE atlas anchors")
    asyncio.run(main())
