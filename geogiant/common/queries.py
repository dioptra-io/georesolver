from loguru import logger
from collections import defaultdict
from pych_client import ClickHouseClient, AsyncClickHouseClient
from pych_client.exceptions import ClickHouseException

from geogiant.clickhouse import (
    GetVPs,
    GetVPsIds,
    GetPingsPerTarget,
    GetPingsPerSrcDst,
    GetDNSMappingHostnames,
    GetDNSMappingPerHostnames,
    GetVPsSubnets,
    GetLastMileDelay,
    InsertFromCSV,
    CreatePingTable,
    CreateScoreTable,
    CreateGeolocTable,
    CreateTracerouteTable,
    GetSubnets,
    GetDstPrefix,
    GetTargetScore,
    GetMeasurementIds,
    GetShortestPingResults,
    GetCachedTargets,
    GetMeasurementIds,
)
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

    try:
        with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
            CreatePingTable().execute(client, table_name)
            resp = GetPingsPerTarget().execute(
                client=client,
                table_name=table_name,
                filtered_vps=removed_vps,
            )

        for row in resp:
            ping_vps_to_target[row["target"]] = row["pings"]

        logger.info(f"Retrived pings for {len(ping_vps_to_target)} targets")

    except ClickHouseException:
        logger.warning(
            f"Something went wrong. Probably that {table_name} does not exists"
        )
        pass

    return ping_vps_to_target


def load_target_geoloc(table_name: str, targets: list[str]) -> dict:
    """
    return shortest ping results for all targets
    """
    targets_geoloc = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetShortestPingResults().execute(
            client=client,
            table_name=table_name,
            filtered_targets=targets,
        )

    for row in resp:
        target_addr = row["dst_addr"]
        vp_addr = row["shortest_ping"][0]
        msm_id = row["shortest_ping"][1]
        min_rtt = row["min_rtt"]

        targets_geoloc[target_addr] = (vp_addr, msm_id, min_rtt)

    return targets_geoloc


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


def load_cached_targets(table_name: str, filtered_targets: list[str] = []) -> list[str]:
    cached_targets = []
    try:
        with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
            resp = GetCachedTargets().execute(
                client=client,
                table_name=table_name,
                filtered_targets=filtered_targets,
            )

        for row in resp:
            cached_targets.append(row["target"])

    except ClickHouseException:
        logger.warning(
            f"Something went wrong. Probably that {table_name} does not exists"
        )
        pass

    return cached_targets


def get_measurement_ids(measurement_table: str) -> set:
    """return all the measurement ids that were saved"""
    measurement_ids = set()
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetMeasurementIds().execute(client, measurement_table)

        for row in resp:
            measurement_ids.add(row["vp_ids"])

    return measurement_ids


def get_vps_ids(ping_table: str) -> set:
    """retrieve all VPs that participated to a measurement in the past"""
    vp_ids_per_target = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetVPsIds().execute(client, ping_table)

        for row in resp:

            vp_ids_per_target[row["dst_addr"]] = row["vp_ids"]

    return vp_ids_per_target


def get_subnets_mapping(
    dns_table: str,
    subnets: list[str],
    hostname_filter: list[str] = None,
    print_error: bool = True,
) -> dict:
    """get ecs-dns resolution per hostname for all input subnets"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = GetDNSMappingHostnames().execute_iter(
            client=client,
            table_name=dns_table,
            subnet_filter=subnets,
            hostname_filter=hostname_filter,
        )

        subnets_mapping = defaultdict(dict)
        try:
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

        except ClickHouseException as e:
            if print_error:
                logger.warning(
                    f"Something went wrong. Probably that {dns_table} does not exists:: {e}"
                )
            pass

    return subnets_mapping


def get_subnets(
    table_name: str,
    subnets: list[str] = [],
    print_error: bool = True,
) -> dict:
    target_subnets = []
    try:
        with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
            rows = GetSubnets().execute(
                client=client, table_name=table_name, subnet_filter=[s for s in subnets]
            )

        target_subnets = [row["subnet"] for row in rows]

    except ClickHouseException as e:
        if print_error:
            logger.warning(
                f"Something went wrong. Probably that {table_name} does not exists:: {e}"
            )
            pass

    return target_subnets


def get_dst_prefix(ping_table: str) -> list[str]:
    """retrieve subnets from ping table"""
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        CreatePingTable().execute(client, ping_table)
        rows = GetDstPrefix().execute(
            client=client,
            table_name=ping_table,
        )

    subnets = []
    for row in rows:
        subnets.append(row["dst_prefix"])

    return subnets


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
            subnet_filter=subnets,
            hostname_filter=hostname_filter,
        )

        mapping_per_hostname = defaultdict(dict)
        for row in resp:
            subnet = row["client_subnet"]
            answer_bgp_prefixes = row["answer_bgp_prefixes"]
            hostname = row["hostname"]

            mapping_per_hostname[hostname][subnet] = answer_bgp_prefixes

    return mapping_per_hostname


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


def load_target_scores(score_table: str, subnets: list[str]) -> dict:
    target_score = {}

    try:
        with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
            resp = GetTargetScore().execute(
                client=client,
                table_name=score_table,
                subnet_filter=subnets,
            )

        for row in resp:
            target_score[row["subnet"]] = row["vps_score"]

    except ClickHouseException as e:
        logger.warning(
            f"Something went wrong. Probably that {score_table} does not exists:: {e}"
        )
        pass

    return target_score


async def insert_pings(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreatePingTable().aio_execute(client=client, table_name=output_table)
        await InsertFromCSV().execute(
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_traceroutes(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreateTracerouteTable().aio_execute(
            client=client, table_name=output_table
        )
        await InsertFromCSV().execute(
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_scores(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreateScoreTable().aio_execute(client=client, table_name=output_table)
        await InsertFromCSV().execute(
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_geoloc(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreateGeolocTable().aio_execute(client=client, table_name=output_table)
        await InsertFromCSV().execute(
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def get_ping_measurement_ids(table_name: str) -> list[int]:
    """return all measurement ids that were already inserted within clickhouse"""
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        await CreatePingTable().aio_execute(client, table_name)
        resp = await GetMeasurementIds().aio_execute(client, table_name)

        measurement_ids = []
        for row in resp:
            measurement_ids.append(row["msm_id"])

        return measurement_ids
