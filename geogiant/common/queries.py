from pych_client import ClickHouseClient
from collections import defaultdict

from geogiant.clickhouse import (
    GetVPs,
    GetPingsPerTarget,
    GetPingsPerSrcDst,
    GetCompleVPs,
)
from geogiant.common.settings import ClickhouseSettings

clickhouse_settings = ClickhouseSettings()


def get_pings_per_target(table_name: str, removed_vps: list = []) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetPingsPerTarget().execute(
            client=client,
            table_name=table_name,
            filtered_vps=removed_vps,
        )

    for row in resp:
        ping_vps_to_target[row["target"]] = row["pings"]

    return ping_vps_to_target


def get_pings_per_src_dst(
    table_name: str, removed_vps: list = [], threshold: int = 300
) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = defaultdict(dict)
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetPingsPerSrcDst().execute(
            client=client,
            table_name=table_name,
            filtered_vps=removed_vps,
            threshold=threshold,
        )

    for row in resp:
        ping_vps_to_target[row["dst"]][row["src"]] = row["min_rtt"]

    return ping_vps_to_target


def load_all_vps(input_table: str) -> list:
    """retrieve all VPs from clickhouse"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = GetCompleVPs().execute(client=client, table_name=input_table)

    return vps


def load_vps(input_table: str) -> list:
    """retrieve all VPs from clickhouse"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = GetVPs().execute(client=client, table_name=input_table)

    return vps


def load_targets(input_table: str) -> list:
    """load all targets (ripe atlas anchors) from clickhouse"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        targets = GetVPs().execute(
            client=client, table_name=input_table, is_anchor=True
        )

    return targets
