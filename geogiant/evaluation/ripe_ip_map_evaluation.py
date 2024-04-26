import asyncio

from loguru import logger
from pych_client import ClickHouseClient


from geogiant.clickhouse import GetDstPrefix
from geogiant.prober.ripe_api import RIPEAtlasAPI
from geogiant.hostname_init import resolve_vps_subnet
from geogiant.common.files_utils import load_csv, load_json, dump_json
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()
table_name = "pings_ripe_ip_map"
output_table = "ripe_ip_map_ecs_mapping"
latency_thresold = 50


async def get_ripe_ip_map_results(tag: str = "single-radius") -> None:
    """get all measurements from RIPE IP map single radius engine"""
    await RIPEAtlasAPI().get_ripe_ip_map_measurements(
        tag=tag, output_table=table_name, max_age_days=5
    )


def get_ripe_ip_map_subnets(latency_thresold: int) -> list:
    """retrieve all RIPE IP map subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetDstPrefix().execute(
            client=client, table_name=table_name, latency_thresold=latency_thresold
        )

    ripe_ip_map_subnets = []
    for row in rows:
        ripe_ip_map_subnets.append(row["dst_prefix"])

    return ripe_ip_map_subnets


async def resolve_subnets() -> None:
    selected_hostnames = load_csv(
        path_settings.DATASET / "selected_hostname_geo_score.csv"
    )

    ripe_ip_map_subnets_path = path_settings.DATASET / "ripe_ip_map_subnets.json"
    if not ripe_ip_map_subnets_path.exists():
        ripe_ip_map_subnets = get_ripe_ip_map_subnets(latency_thresold)

        logger.info(
            f"Retrieved:: {len(ripe_ip_map_subnets)} RIPE IP map subnets, {latency_thresold=}"
        )
        dump_json(ripe_ip_map_subnets, ripe_ip_map_subnets_path)

    ripe_ip_map_subnets = load_json(ripe_ip_map_subnets_path)

    await resolve_vps_subnet(
        selected_hostnames=selected_hostnames,
        input_file=ripe_ip_map_subnets_path,
        output_table=output_table,
        chunk_size=500,
    )


async def main() -> None:
    # retrieve all IP addresses with < 2ms
    # analysis
    # score
    # Ping schedule + measurement
    retrieve_public_measurements = False
    make_measurement = True

    if retrieve_public_measurements:
        await get_ripe_ip_map_results()

    if resolve_subnets:
        await resolve_subnets()


if __name__ == "__main__":
    asyncio.run(main())
