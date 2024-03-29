from collections import defaultdict
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import (
    GetDNSMappingHostnames,
    GetPoPInfo,
    GetPoPPerHostname,
    GetPingsPerTarget,
    GetTargets,
    GetVPs,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def get_pings_per_target(table_name: str) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetPingsPerTarget().aio_execute(
            client=client,
            table_name=table_name,
        )

    for row in resp:
        ping_vps_to_target[row["target"]] = row["pings"]

    return ping_vps_to_target


async def get_subnets_mapping(
    dns_table: str,
    subnets: list[str],
    hostname_filter: list[str] = None,
) -> dict:
    """get ecs-dns resolution per hostname for all input subnets"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await GetDNSMappingHostnames().execute_iter(
            client=client,
            table_name=dns_table,
            subnet_filter=[s for s in subnets],
            hostname_filter=hostname_filter,
        )

        subnets_mapping = defaultdict(dict)
        async for row in resp:
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


async def get_pop_info(
    dns_table: str,
    answer_granularity: str,
    hostname_filter: list[str] = None,
) -> dict:
    """get all geolocated PoP information"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:

        resp = await GetPoPInfo().execute_iter(
            client=client,
            table_name=dns_table,
            hostname_filter=hostname_filter,
        )

        pop_info = {}
        async for row in resp:
            pop = row[answer_granularity]
            del row[answer_granularity]
            pop_info[pop] = row

    return pop_info


async def get_pop_per_hostname(
    dns_table: str,
    hostname_filter: list[str] = None,
) -> dict:
    """get all geolocated PoP information"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:

        resp = await GetPoPPerHostname().execute_iter(
            client=client,
            table_name=dns_table,
            hostname_filter=hostname_filter,
        )

        pop_per_hostname = defaultdict(dict)
        async for row in resp:
            hostname = row["hostname"]

            for subnet, lat, lon in row["pop"]:

                pop_per_hostname[hostname][subnet] = (lat, lon)

    return pop_per_hostname


async def get_subnet_per_pop(
    dns_table: str,
    hostname_filter: list[str] = None,
) -> dict:
    """get all subnets per PoP (i.e. subnets that share the same geoloc)"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:

        resp = await GetPoPPerHostname().execute_iter(
            client=client,
            table_name=dns_table,
            hostname_filter=hostname_filter,
        )

        pop_per_hostname = defaultdict(dict)
        async for row in resp:
            hostname = row["hostname"]

            for subnet, lat, lon in row["pop"]:
                try:
                    pop_per_hostname[hostname][(lat, lon)].append(subnet)
                except KeyError:
                    pop_per_hostname[hostname][(lat, lon)] = [subnet]

    return pop_per_hostname


async def load_targets(dns_table: str) -> dict:
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        targets = await GetTargets().aio_execute(
            client=client,
            table_name=dns_table,
        )

    return targets


async def load_vps(dns_table: str) -> dict:
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = await GetVPs().aio_execute(
            client=client,
            table_name=dns_table,
        )

        return vps
