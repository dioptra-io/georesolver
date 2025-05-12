from loguru import logger
from typing import Generator, Any
from collections import defaultdict
from pych_client import ClickHouseClient, AsyncClickHouseClient
from pych_client.exceptions import ClickHouseException

from georesolver.clickhouse import (
    Query,
    CreatePingTable,
    CreateScoreTable,
    CreateScheduleTable,
    CreateGeolocTable,
    CreateDNSMappingTable,
    CreateTracerouteTable,
    CreateNameServerTable,
    GetTables,
    ChangeTableName,
    GetSubnets,
    GetTargets,
    GetMeasurementIds,
    GetLastMileDelay,
    GetLastMileDelayPerId,
    GetCachedTargets,
    GetVPs,
    GetIPv6VPs,
    GetVPsIds,
    GetVPsSubnets,
    GetPingsPerSrcDst,
    GetPingsPerTarget,
    GetPingsPerVP,
    GetPingsPerTargetExtended,
    GetShortestPingResults,
    GetECSResults,
    GetHostnames,
    GetDNSMappingHostnames,
    GetDNSMappingPerHostnames,
    GetDNSMappingAnswers,
    GetAnswersPerHostname,
    GetTargetScore,
)
from georesolver.common.files_utils import create_tmp_csv_file
from georesolver.common.settings import ClickhouseSettings


def get_tables() -> list[str]:
    tables = []

    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetTables().execute(client=client, table_name="")

    for row in resp:
        tables.append(row["name"])

    return tables


def change_table_name(table_name: str, new_table_name: str) -> None:
    """execute change table name query"""
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        # check if out table does not already exists
        tables = get_tables()
        if new_table_name in tables:
            logger.warning(
                f"Output table:: {new_table_name} already exists, skipping step"
            )
            return
        if table_name not in tables:
            logger.warning(f"Output table:: {table_name} already exists, skipping step")
            return

        ChangeTableName().execute(
            client=client,
            table_name=table_name,
            new_table_name=new_table_name,
        )


def get_min_rtt_per_vp_per_id(table_name: str) -> dict:
    """get the minimum RTT per VP from 50 traceroutes samples"""
    last_mile_delay_per_vp = {}
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetLastMileDelayPerId().execute(
            client=client,
            table_name=table_name,
        )

    for row in resp:
        last_mile_delay_per_vp[row["prb_id"]] = row["min_rtt"]

    return last_mile_delay_per_vp


def get_min_rtt_per_vp(table_name: str) -> dict:
    """get the minimum RTT per VP from 50 traceroutes samples"""
    last_mile_delay_per_vp = {}
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetLastMileDelay().execute(
            client=client,
            table_name=table_name,
        )

    for row in resp:
        last_mile_delay_per_vp[row["vp_addr"]] = row["min_rtt"]

    return last_mile_delay_per_vp


def get_pings_per_target(
    table_name: str, removed_vps: list = [], nb_packets: int = -1
) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}

    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            resp = GetPingsPerTarget().execute(
                client=client,
                table_name=table_name,
                filtered_vps=removed_vps,
                nb_packets=nb_packets,
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


def get_pings_per_target_extended(
    table_name: str, removed_vps: list = [], latency_threshold: int = 300
) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
    """
    ping_vps_to_target = {}

    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            resp = GetPingsPerTargetExtended().execute_iter(
                client=client,
                table_name=table_name,
                filtered_vps=removed_vps,
                latency_threshold=latency_threshold,
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


def get_targets(table_name: str, threshold: int = 300) -> int:
    """count the number of targets from a table"""
    targets = []
    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            resp = GetTargets().execute_iter(
                client=client, table_name=table_name, threshold=threshold
            )

            for row in resp:
                targets.append(row["target"])

            return targets

    except ClickHouseException:
        logger.warning(
            f"Something went wrong. Probably that {table_name} does not exists"
        )
        pass


def iter_pings_per_target(
    table_name: str,
    removed_vps: list = [],
    threshold: int = 300,
) -> Generator[Any, Any, list[dict]]:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr,prb_id , min_rtt)]
    """

    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            resp = GetPingsPerTargetExtended().execute_iter(
                client=client,
                table_name=table_name,
                filtered_vps=removed_vps,
                threshold=threshold,
            )

            for row in resp:
                yield row

    except ClickHouseException:
        logger.warning(
            f"Something went wrong. Probably that {table_name} does not exists"
        )
        pass


def iter_pings_per_vp(
    table_name: str,
    removed_vps: list = [],
    threshold: int = 300,
) -> Generator[Any, Any, list[dict]]:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_addr,prb_id , min_rtt)]
    """

    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
            resp = GetPingsPerVP().execute_iter(
                client=client,
                table_name=table_name,
                filtered_vps=removed_vps,
                threshold=threshold,
            )

            for row in resp:
                yield row

    except ClickHouseException:
        logger.warning(
            f"Something went wrong. Probably that {table_name} does not exists"
        )
        pass


def load_target_geoloc(table_name: str, removed_vps: list[str] = []) -> dict:
    """
    return shortest ping results for all targets
    """
    targets_geoloc = {}
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        try:
            resp = GetShortestPingResults().execute(
                client=client,
                table_name=table_name,
                removed_vps=removed_vps,
            )
        except:
            logger.error(f"Table:: {table_name} does not exists")
            return None

    for row in resp:
        target_addr = row["dst_addr"]
        vp_addr = row["shortest_ping"][0]
        prb_id = row["shortest_ping"][1]
        min_rtt = row["min_rtt"]

        targets_geoloc[target_addr] = (vp_addr, prb_id, min_rtt)

    return targets_geoloc


def get_pings_per_src_dst(
    table_name: str, removed_vps: list = [], threshold: int = 300
) -> dict:
    """
    return meshed ping for all targets
    ping_vps_to_target[target_addr] = [(vp_id, min_rtt)]
    """
    ping_vps_to_target = defaultdict(dict)
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetPingsPerSrcDst().execute(
            client=client,
            table_name=table_name,
            filtered_vps=removed_vps,
            threshold=threshold,
        )

    for row in resp:
        ping_vps_to_target[row["dst"]][row["prb_id"]] = row["min_rtt"]

    return ping_vps_to_target


def load_vps(input_table: str, ipv6: bool = False) -> list:
    """retrieve all VPs from clickhouse"""
    vps = []
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        if not ipv6:
            rows = GetVPs().execute(client=client, table_name=input_table)
        else:
            rows = GetIPv6VPs().execute(client=client, table_name=input_table)

    vps_id = set()
    for row in rows:
        if row["id"] in vps_id:
            continue
        else:
            vps.append(row)
            vps_id.add(row["id"])

    return vps


def load_targets(input_table: str) -> list:
    """load all targets (ripe atlas anchors) from clickhouse"""
    targets = []
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        rows = GetVPs().execute(client=client, table_name=input_table, is_anchor=True)

        target_ids = set()
        for row in rows:
            if row["id"] in target_ids:
                continue
            else:
                targets.append(row)
                target_ids.add(row["id"])

    return targets


def load_cached_targets(table_name: str, filtered_targets: list[str] = []) -> list[str]:
    cached_targets = []
    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
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
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:

        try:
            resp = GetMeasurementIds().execute(client, measurement_table)

            for row in resp:
                measurement_ids.add(row["msm_id"])
        except ClickHouseException:
            logger.warning(f"Table {measurement_table} does not exists")
            pass

    return measurement_ids


def get_vps_ids_per_target(ping_table: str) -> set:
    """retrieve all VPs that participated to a measurement in the past"""
    vp_ids_per_target = {}
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetVPsIds().execute(client, ping_table)

        for row in resp:

            vp_ids_per_target[row["dst_addr"]] = row["vp_ids"]

    return vp_ids_per_target


def get_subnets_mapping(
    dns_table: str,
    subnets: list[str] = [],
    hostname_filter: list[str] = None,
    print_error: bool = True,
) -> dict[dict]:
    """get ecs-dns resolution per hostname for all input subnets"""
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
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
                answer_asn = row["answer_asns"]
                hostname = row["hostname"]
                source_scope = row["source_scope"]

                subnets_mapping[subnet][hostname] = {
                    "answers": answers,
                    "answer_subnets": answer_subnets,
                    "answer_bgp_prefixes": answer_bgp_prefixes,
                    "answer_asns": answer_asn,
                    "source_scope": source_scope,
                }

        except ClickHouseException as e:
            if print_error:
                logger.warning(
                    f"Something went wrong. Probably that {dns_table} does not exists:: {e}"
                )
            pass

    return subnets_mapping


def get_mapping_answers(dns_table: str) -> list[str]:
    """get all IP addresses from a table of DNS mapping"""
    answers = set()
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetDNSMappingAnswers().execute_iter(client=client, table_name=dns_table)

        for row in resp:
            answers.add(row["answer"])

    return list(answers)


def get_answers_per_hostname(dns_table: str) -> dict[list[str]]:
    """get all IP addresses from a table of DNS mapping"""
    answers_per_hostname = {}
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetAnswersPerHostname().execute_iter(client=client, table_name=dns_table)

        for row in resp:
            hostname = row["hostname"]
            answers = row["answers"]
            answers_per_hostname[hostname] = answers

    return answers_per_hostname


def get_ecs_results(
    dns_table: str,
    subnets: list[str] = [],
    hostname_filter: list[str] = None,
    print_error: bool = True,
) -> dict[dict[list[str]]]:
    """get ecs-dns resolution per hostname for all input subnets"""
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetECSResults().execute_iter(
            client=client,
            table_name=dns_table,
            subnet_filter=subnets,
            hostname_filter=hostname_filter,
        )

        subnets_mapping = defaultdict(dict)
        try:
            for row in resp:
                subnet = row["client_subnet"]
                hostname = row["hostname"]
                answer_subnets = row["answer_subnets"]

                subnets_mapping[subnet][hostname] = answer_subnets

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
) -> list[str]:
    """retrieve all distinct /24 prefixes from table_name"""
    target_subnets = []
    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
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


def get_hostnames(dns_table: str) -> list[str]:
    """retrieve all hostnames from DNS Mapping table"""
    hostnames = []
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = GetHostnames().execute_iter(client=client, table_name=dns_table)
        for row in resp:
            hostnames.append(row["hostname"])

    return hostnames


def get_mapping_per_hostname(
    dns_table: str,
    subnets: list[str],
    hostname_filter: list[str] = None,
) -> dict:
    """get ecs-dns resolution per hostname for all input subnets"""
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
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
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        targets = GetVPsSubnets().execute(
            client=client, table_name=dns_table, is_anchor=True
        )

    target_subnets = [target["subnet"] for target in targets]

    return target_subnets


def load_vp_subnets(dns_table: str) -> dict:
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        vps = GetVPsSubnets().execute(
            client=client,
            table_name=dns_table,
        )

    vp_subnets = [vp["subnet"] for vp in vps]

    return vp_subnets


def load_target_scores(score_table: str, subnets: list[str]) -> dict:
    target_score = {}

    try:
        with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
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


def insert_dns_answers(
    csv_data: list[str],
    output_table: str,
    request_type: str,
) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        if request_type == "A":
            CreateDNSMappingTable().execute(client=client, table_name=output_table)
        elif request_type == "NS":
            CreateNameServerTable().execute(client=client, table_name=output_table)

        Query().execute_insert(
            client=client,
            table_name=output_table,
            in_file=tmp_file_path,
        )


async def insert_pings(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        await CreatePingTable().aio_execute(client=client, table_name=output_table)
        Query().execute_insert(
            client=client,
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_traceroutes(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        await CreateTracerouteTable().aio_execute(
            client=client, table_name=output_table
        )
        Query().execute_insert(
            client=client,
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_schedule(csv_data: list[str], output_table: str) -> None:
    """insert target schedule into output table"""
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        await CreateScheduleTable().aio_execute(client=client, table_name=output_table)
        Query().execute_insert(
            client=client,
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_scores(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        await CreateScoreTable().aio_execute(client=client, table_name=output_table)
        Query().execute_insert(
            client=client,
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def insert_geoloc(csv_data: list[str], output_table: str) -> None:
    tmp_file_path = create_tmp_csv_file(csv_data)

    async with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        await CreateGeolocTable().aio_execute(client=client, table_name=output_table)
        Query().execute_insert(
            client=client,
            table_name=output_table,
            in_file=tmp_file_path,
        )

    tmp_file_path.unlink()


async def get_ping_measurement_ids(table_name: str) -> list[int]:
    """return all measurement ids that were already inserted within clickhouse"""
    async with AsyncClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        resp = await GetMeasurementIds().aio_execute(client, table_name)

        measurement_ids = []
        for row in resp:
            measurement_ids.append(row["msm_id"])

        return measurement_ids


def change_table_name(table_name: str, new_table_name: str) -> None:
    """execute change table name query"""
    with ClickHouseClient(**ClickhouseSettings().clickhouse) as client:
        # check if out table does not already exists
        tables = get_tables()
        if table_name not in tables:
            logger.warning(
                f"Old table {table_name} did not exists, no need for name change"
            )
            return
        if new_table_name in tables:
            logger.warning(
                f"Output table:: {new_table_name} already exists, skipping step"
            )
        else:
            ChangeTableName().execute(
                client=client,
                table_name=table_name,
                new_table_name=new_table_name,
            )
