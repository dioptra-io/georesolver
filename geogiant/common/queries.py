import time

from tqdm import tqdm
from collections import defaultdict
from pych_client import ClickHouseClient, AsyncClickHouseClient

from geogiant.clickhouse import (
    GetVPs,
    GetPingsPerTarget,
    GetPingsPerSrcDst,
    GetCompleVPs,
    GetDNSMappingHostnames,
    GetDNSMappingPerHostnames,
    GetVPsSubnets,
    GetLastMileDelay,
    InsertFromCSV,
    CreatePingTable,
    GetSubnets,
)
from geogiant.prober.ripe_api import RIPEAtlasAPI
from geogiant.common.files_utils import create_tmp_csv_file
from geogiant.common.settings import ClickhouseSettings

clickhouse_settings = ClickhouseSettings()


def get_min_rtt_per_vp(table_name: str) -> dict:
    """get the minimum RTT per VP from 50 traceroutes samples"""
    last_mile_delay_per_vp = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetLastMileDelay().execute(
            client=client,
            table_name=table_name,
        )

    for row in resp:
        last_mile_delay_per_vp[row["src_addr"]] = row["min_rtt"]

    return last_mile_delay_per_vp


def get_pings_per_target(table_name: str, removed_vps: list = []) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        CreatePingTable().execute(client, table_name)
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


def load_subnets_from_ecs_mapping() -> list:
    """retrieve all RIPE IP map subnets"""
    # get routers 2ms subnets
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetSubnets().execute(client=client, table_name=ECS_TABLE)

    end_to_end_subnets = []
    for row in rows:
        end_to_end_subnets.append(row["subnet"])

    return end_to_end_subnets


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


def get_subnets_mapping(
    dns_table: str,
    subnets: list[str],
    hostname_filter: list[str] = None,
) -> dict:
    """get ecs-dns resolution per hostname for all input subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetDNSMappingHostnames().execute_iter(
            client=client,
            table_name=dns_table,
            subnet_filter=[s for s in subnets],
            hostname_filter=hostname_filter,
        )

        subnets_mapping = defaultdict(dict)
        for row in resp:
            subnet = row["client_subnet"]
            answers = row["answers"]
            answer_subnets = row["answer_subnets"]
            answer_bgp_prefixes = row["answer_bgp_prefixes"]
            hostname = row["hostname"]
            source_scope = row["source_scope"]

            subnets_mapping[subnet][hostname] = {
                "answers": answers,
                "answer_subnets": answer_subnets,
                "answer_bgp_prefixes": answer_bgp_prefixes,
                "source_scope": source_scope,
            }

    return subnets_mapping


def get_mapping_per_hostname(
    dns_table: str,
    subnets: list[str],
    hostname_filter: list[str] = None,
) -> dict:
    """get ecs-dns resolution per hostname for all input subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetDNSMappingPerHostnames().execute_iter(
            client=client,
            table_name=dns_table,
            subnet_filter=[s for s in subnets],
            hostname_filter=hostname_filter,
        )

        mapping_per_hostname = defaultdict(dict)
        for row in resp:
            subnet = row["client_subnet"]
            answer_bgp_prefixes = row["answer_bgp_prefixes"]
            hostname = row["hostname"]

            mapping_per_hostname[hostname][subnet] = answer_bgp_prefixes

    return mapping_per_hostname


def get_subnets(dns_table: str) -> dict:
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        targets = GetSubnets().execute(client=client, table_name=dns_table)

    target_subnets = [target["subnet"] for target in targets]

    return target_subnets


def load_target_subnets(dns_table: str) -> dict:
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        targets = GetVPsSubnets().execute(
            client=client, table_name=dns_table, is_anchor=True
        )

    target_subnets = [target["subnet"] for target in targets]

    return target_subnets


def load_vp_subnets(dns_table: str) -> dict:
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = GetVPsSubnets().execute(
            client=client,
            table_name=dns_table,
        )

    vp_subnets = [vp["subnet"] for vp in vps]

    return vp_subnets


async def insert_pings(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreatePingTable().aio_execute(client=client, table_name=output_table)
        await InsertFromCSV().execute(
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def retrieve_pings(ids: list[int], output_table: str) -> None:
    """retrieve all ping measurements from a list of measurement ids"""
    csv_data = []
    for id in tqdm(ids):
        ping_results = RIPEAtlasAPI().get_ping_results(id)
        csv_data.extend(ping_results)

        time.sleep(0.1)

    await insert_pings(csv_data, output_table)


async def retrieve_pings_from_tag(tag, output_table: str) -> None:
    """retrieve all results for a specific tag and insert data"""
    csv_data = await RIPEAtlasAPI().get_measurements_from_tag(tag)

    await insert_pings(csv_data, output_table)
