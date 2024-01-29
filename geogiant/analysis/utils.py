from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import GetPingsPerTarget
from geogiant.common.settings import ClickhouseSettings

clickhouse_settings = ClickhouseSettings()


async def get_pings_per_target(table_name: str) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetPingsPerTarget().execute(
            client=client,
            table_name=table_name,
        )

    for row in resp:
        ping_vps_to_target[row["target"]] = row["pings"]

    return ping_vps_to_target
