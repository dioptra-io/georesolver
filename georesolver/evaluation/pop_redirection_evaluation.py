"""study and evaluate CDNs redirection strategies"""

import json
import asyncio
import numpy as np

from tqdm import tqdm
from pyasn import pyasn
from pathlib import Path
from loguru import logger
from random import choice
from collections import defaultdict
from datetime import datetime, timedelta

from georesolver.clickhouse.queries import (
    load_vps,
    get_tables,
    get_ecs_results,
    load_target_geoloc,
    get_measurement_ids,
    get_mapping_answers,
    get_answers_per_hostname,
    get_pings_per_target_extended,
)
from georesolver.zdns.zmap import zmap
from georesolver.agent.ecs_process import run_dns_mapping
from georesolver.agent.insert_process import retrieve_pings
from georesolver.prober import RIPEAtlasProber, RIPEAtlasAPI
from georesolver.evaluation.evaluation_georesolver_functions import (
    get_scores,
    get_vp_selection_per_target,
)
from georesolver.evaluation.evaluation_plot_functions import (
    ecdf,
    plot_cdf,
    plot_multiple_cdf,
    get_proportion_under,
)
from georesolver.common.ip_addresses_utils import (
    get_prefix_from_ip,
    route_view_bgp_prefix,
)
from georesolver.common.geoloc import is_within_cirle
from georesolver.common.files_utils import load_csv, dump_csv, load_json, dump_json
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
ripe_atlas_settings = RIPEAtlasSettings()

MEASURMENT_TAG = "meshed-cdns-pings-test"
PING_TABLE = "meshed_cdns_pings_test"
ECS_TABLE = "meshed_cdns_ecs"
RESULTS_PATH = path_settings.RESULTS_PATH / "pop_redirection_eval"


def get_responsive_answers(answers: list[str], output_path: Path) -> list[str]:
    """get responsive IP addresses from a list of pings"""

    if not output_path.exists():
        subnets = list(set([get_prefix_from_ip(answer) for answer in answers]))
        responsive_answers = zmap(subnets)

        dump_csv(responsive_answers, output_path)

        return responsive_answers

    responsive_answers = load_csv(output_path)

    return responsive_answers


def get_measurement_schedule() -> list[tuple[str, list]]:
    """perform meshed pings towards all CDNs replicas present in VPs mapping"""
    responsive_answers_path = path_settings.DATASET / "responsive_answers_cdns.csv"

    # 1. load all VPs mapping answers
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    answers = get_mapping_answers(ch_settings.VPS_ECS_MAPPING_TABLE)
    all_subnets = set([get_prefix_from_ip(answer) for answer in answers])

    logger.info(f"All answers :: {len(answers)}")
    logger.info(f"All subnets :: {len(all_subnets)}")

    # 2. filter based on responsive answers
    responsive_answers = get_responsive_answers(answers, responsive_answers_path)
    answers = set(answers).intersection(set(responsive_answers))

    logger.info(f"Responsive answers :: {len(answers)}")

    # 3. get answers per subnets
    answer_per_subnets = defaultdict(set)
    for answer in answers:
        subnet = get_prefix_from_ip(answer)
        answer_per_subnets[subnet].add(answer)

    logger.info(f"Total number of subnets to probe {len(all_subnets)}")

    # 3. select one IP address per /24
    measurement_schedule = []
    all_targets = set()
    for subnet, answers in answer_per_subnets.items():
        # select one IP randomly
        target = choice(list(answers))

        # prepare schedule
        batch_size = ripe_atlas_settings.MAX_VP
        for i in range(0, len(vps), batch_size):
            batch_vps = [vp["id"] for vp in vps[i : i + batch_size]]
            measurement_schedule.append((target, batch_vps))

        all_targets.add(target)

    logger.info(f"Total Nb measurement to run:: {len(measurement_schedule)}")

    return measurement_schedule


async def insert_measurements(
    measurement_schedule: list[tuple],
    probing_tags: list[str],
    output_table: str,
    wait_time: int = 60,
) -> None:
    """insert measurement once they are tagged as Finished on RIPE Atlas"""
    current_time = datetime.timestamp(datetime.now() - timedelta(days=2))
    cached_measurement_ids = set()
    while True:
        # load measurement finished from RIPE Atlas
        stopped_measurement_ids = await RIPEAtlasAPI().get_stopped_measurement_ids(
            start_time=current_time, tags=probing_tags
        )

        # load already inserted measurement ids
        inserted_ids = get_measurement_ids(output_table)

        # stop measurement once all measurement are inserted
        all_measurement_ids = set(inserted_ids).union(cached_measurement_ids)
        if len(all_measurement_ids) >= len(measurement_schedule):
            logger.info(
                f"All measurement inserted:: {len(inserted_ids)=}; {len(measurement_schedule)=}"
            )
            break

        measurement_to_insert = set(stopped_measurement_ids).difference(
            set(inserted_ids)
        )

        # check cached measurements,
        # some measurement are not insersed because no results
        measurement_to_insert = set(measurement_to_insert).difference(
            cached_measurement_ids
        )

        logger.info(f"{len(stopped_measurement_ids)=}")
        logger.info(f"{len(inserted_ids)=}")
        logger.info(f"{len(measurement_to_insert)=}")

        if not measurement_to_insert:
            await asyncio.sleep(wait_time)
            continue

        # insert measurement
        batch_size = 1_000
        for i in range(0, len(measurement_to_insert), batch_size):
            logger.info(
                f"Batch {i // batch_size}/{len(measurement_to_insert) // batch_size}"
            )
            batch_measurement_ids = list(measurement_to_insert)[i : i + batch_size]
            await retrieve_pings(batch_measurement_ids, output_table)

        cached_measurement_ids.update(measurement_to_insert)
        current_time = datetime.timestamp((datetime.now()) - timedelta(days=1))

        await asyncio.sleep(wait_time)


async def meshed_ping_cdns(prev_schedule: Path = None) -> None:
    """
    perform pings towards one IP address
    for each /24 prefix present in VPs ECS mapping redirection
    """

    if not prev_schedule:
        measurement_schedule = get_measurement_schedule()
    else:
        measurement_schedule = []

        # filter based on existing schedule/meaasurement
        cached_measurement_schedule = load_json(prev_schedule)
        # get the minimum of scheduled vps for a measurement
        min_nb_vps = len(min(cached_measurement_schedule, key=lambda x: len(x[-1]))[-1])
        # load existing measurement
        cached_measurements = get_pings_per_target_extended(PING_TABLE)

        logger.info(f"{len(cached_measurements)=}")

        for target, vp_ids in tqdm(cached_measurement_schedule):
            # retrieve pings for target, if exists
            try:
                cached_ping = cached_measurements[target]
            except KeyError:
                continue

            # only keep vp ids not in cached measurements
            cached_vp_ids = [vp_id for _, vp_id, _ in cached_ping]
            remaning_vp_ids = list(set(vp_ids).difference(cached_vp_ids))

            measurement_schedule.append((target, remaning_vp_ids))

        # filter targets for which measurement was done, but not all vps responded
        filtered_measurement_schedule = []
        logger.info(f"Filtering measurements under min nb vp ids:: {min_nb_vps}")
        for target, vp_ids in measurement_schedule:
            if len(vp_ids) < min_nb_vps:
                continue

            filtered_measurement_schedule.append((target, vp_ids))

        measurement_schedule = filtered_measurement_schedule

        logger.info(f"{len(measurement_schedule)=}")

    prober = RIPEAtlasProber(
        probing_type="ping",
        probing_tag=MEASURMENT_TAG,
        output_table=PING_TABLE,
        protocol="ICMP",
    )

    await asyncio.gather(
        prober.main(measurement_schedule),
        insert_measurements(
            measurement_schedule,
            probing_tags=["dioptra", MEASURMENT_TAG],
            output_table=PING_TABLE,
        ),
    )

    logger.info("Meshed CDNs pings measurement done")


def compute_percentile_rank(redirected_latency, all_latencies):
    sorted_latencies = sorted(all_latencies)
    # Percentile rank: proportion of latencies less than or equal to redirected
    rank = np.searchsorted(sorted_latencies, redirected_latency, side="right") / len(
        sorted_latencies
    )
    return rank * 100  # in percent


def latency_eval() -> None:
    """evaluate if the redirected PoP was the most optimal based on ping measurements"""
    # 1. load data
    vps_ecs_mapping = get_ecs_results(ch_settings.VPS_ECS_MAPPING_TABLE)
    answers_per_hostname = get_answers_per_hostname(ch_settings.VPS_ECS_MAPPING_TABLE)
    pings_per_target = get_pings_per_target_extended(PING_TABLE)

    # get subnets answer to hostaname
    hostname_per_answer_subnet = {}
    for hostname, answers in answers_per_hostname.items():
        for answer in answers:
            answer_subnet = get_prefix_from_ip(answer)
            hostname_per_answer_subnet[answer_subnet] = hostname

    # 2. get ping per vp to target subnet
    pings_per_vp_hostname = defaultdict(dict)
    pings_per_vp = defaultdict(dict)
    for target_addr, pings in pings_per_target.items():
        target_subnet = get_prefix_from_ip(target_addr)
        hostname = hostname_per_answer_subnet[target_subnet]
        for vp_addr, _, min_rtt in pings:
            try:
                pings_per_vp[vp_addr][target_subnet].append((target_addr, min_rtt))
            except KeyError:
                pings_per_vp[vp_addr][target_subnet] = [(target_addr, min_rtt)]
            try:
                pings_per_vp_hostname[vp_addr][hostname].append(min_rtt)
            except KeyError:
                pings_per_vp_hostname[vp_addr][hostname] = [min_rtt]

    # 3. get redirection ping and calculate rank per pair (hostname; vp)
    percentile_per_hostname = defaultdict(list)
    for vp_addr, pings_per_hostname in tqdm(pings_per_vp_hostname.items()):
        for hostname, all_hostname_pings in pings_per_hostname.items():
            try:
                # retrieved all answer subnets, if exists
                answer_subnets = vps_ecs_mapping[get_prefix_from_ip(vp_addr)][hostname]
            except KeyError:
                continue
            for answer_subnet in answer_subnets:
                try:
                    # get redirection ping, if exists
                    redirection_pings = pings_per_vp[vp_addr][answer_subnet]
                    for _, min_rtt in redirection_pings:
                        # get rank of redirection ping
                        percentile = compute_percentile_rank(
                            min_rtt, all_hostname_pings
                        )
                        # save percentile per hostname
                        percentile_per_hostname[hostname].append(percentile)

                except KeyError:
                    continue

    print(f"{len(percentile_per_hostname)=}")

    # get avg percentile per hostname
    avg_percentiles = []
    for hostname, percentiles in percentile_per_hostname.items():
        avg_percentiles.append(np.average(percentiles))

    x, y = ecdf(avg_percentiles)
    plot_cdf(
        x=x,
        y=y,
        x_lim_left=0,
        output_path="cdf_avg_percentile_latency_per_hostname",
        x_label="avg latency percentile \n accross VPs subnet",
        y_label="fraction of hostnames",
    )


def load_as_to_org(input_path: Path, providers: list[str] = None) -> dict:
    """parse original CAIDA txt file to a dict, return ASN to org data"""
    if not input_path.exists():
        # load raw file
        raw_data_path = path_settings.STATIC_FILES / "20250401.as-org2info.jsonl"
        asn_to_org_name = {}
        with raw_data_path.open("r") as f:
            for row in tqdm(f.readlines()):
                row = json.loads(row)
                # get org info
                if "asn" in row and "name" in row:
                    try:
                        asn = row["asn"]
                        org_name = row["name"]

                        # parse main CDNs orgs (ex: GOOGLER-FIBER -> GOOGLE)
                        for provider in providers:
                            if provider in org_name:
                                org_name = provider

                        asn_to_org_name[asn] = org_name
                    except KeyError as e:
                        logger.error(f"{row}")
                        raise RuntimeError(f"{e}")

        # save data
        dump_json(asn_to_org_name, input_path)
        return asn_to_org_name

    asn_to_org_name = load_json(input_path)
    return asn_to_org_name


def geo_eval() -> None:
    """check if the redirected PoP was the closest one or not"""
    rows_providers_geoloc = load_csv(
        path_settings.STATIC_FILES / "providers_geoloc.csv"
    )
    providers_geoloc = defaultdict(dict)
    for row in rows_providers_geoloc[1:]:
        row = row.split(",")
        providers_geoloc[row[0].upper()][row[1]] = {
            "lat": row[2],
            "lon": row[3],
        }

    asndb = pyasn(str(path_settings.STATIC_FILES / "rib_table_2025_04_24.dat"))
    asn_to_org = load_as_to_org(
        path_settings.STATIC_FILES / "asn_to_org_name.json", providers_geoloc.keys()
    )
    asn_to_org = {int(asn): org_name for asn, org_name in asn_to_org.items()}

    # load DNS mapping data
    answers = get_mapping_answers(ch_settings.VPS_ECS_MAPPING_TABLE)
    # get answer org
    org_per_answer = {}
    for answer in answers:
        # get asn
        asn, _ = route_view_bgp_prefix(answer, asndb)
        # get org name
        try:
            org_name = asn_to_org[asn]
            org_per_answer[answer] = org_name
        except KeyError:
            continue

    # get pings per target and vps
    removed_vps = load_json(path_settings.REMOVED_VPS)
    pop_geoloc = load_target_geoloc(PING_TABLE, removed_vps)
    vps = load_vps(ch_settings.VPS_FILTERED_TABLE)
    vps_per_id = {}
    for vp in vps:
        vps_per_id[vp["id"]] = vp

    candidates_pop_per_target = defaultdict(list)
    missing_orgs = set()
    missing_provider = set()
    missing_pop = set()
    under_2ms = set()
    logger.info(f"{len(pop_geoloc)=}")
    for target_addr, (_, vp_id, min_rtt) in pop_geoloc.items():
        if min_rtt < 2:
            under_2ms.add(target_addr)
        try:
            # get org for target, if exists
            org_target = org_per_answer[target_addr]
            # get provider geoloc, if exists
            provider_geoloc: dict = providers_geoloc[org_target]
        except KeyError:
            missing_orgs.add(target_addr)
            continue

        # get vp_lat, vp_lon
        vp = vps_per_id[vp_id]
        candidates = []
        # check if pop geolocs are within area of presence around vp
        for code, geoloc in provider_geoloc.items():
            pop_lat, pop_lon = float(geoloc["lat"]), float(geoloc["lon"])
            if is_within_cirle((vp["lat"], vp["lon"]), min_rtt, (pop_lat, pop_lon)):
                candidates.append(
                    {
                        "provider": org_target,
                        "code": code,
                        "geoloc": geoloc,
                        "min_rtt": min_rtt,
                        "vp": vp,
                    }
                )

        if not provider_geoloc:
            missing_provider.add(target_addr)
            if not candidates:
                missing_pop.add(target_addr)

            continue

        candidates_pop_per_target[target_addr] = candidates

    logger.info(f"{len(missing_orgs)=}")
    logger.info(f"{len(under_2ms)=}")
    logger.info(f"{len(missing_provider)=}")
    logger.info(f"{len(missing_pop)=}")
    logger.info(f"{len(candidates_pop_per_target)=}")

    dump_json(
        candidates_pop_per_target, RESULTS_PATH / "candidates_pop_per_target.json"
    )


async def ecs_cdns_answers() -> None:
    """perform GeoResolver ECS-DNS resolution on VPs mapping answers"""
    answers = get_mapping_answers(ch_settings.VPS_ECS_MAPPING_TABLE)
    answer_subnets = list(set([get_prefix_from_ip(a) for a in answers]))

    await run_dns_mapping(
        subnets=answer_subnets,
        hostname_file=path_settings.HOSTNAMES_GEORESOLVER,
        output_table="meshed_cdns_ecs",
        itterative=False,
    )


def cdn_meshed_vs_georesolver() -> None:
    """plot latency results between meshed pings and georesolver on CDNs IP addresses"""
    tables = get_tables()
    if ECS_TABLE not in tables:
        asyncio.run(ecs_cdns_answers())

    removed_vps = load_json(path_settings.REMOVED_VPS)
    vps = load_vps(ch_settings.VPS_FILTERED_FINAL_TABLE)
    pings_per_target = get_pings_per_target_extended(PING_TABLE, removed_vps)

    # load score similarity between vps and targets
    scores = get_scores(
        output_path=RESULTS_PATH / "score.pickle",
        hostnames=load_csv(path_settings.HOSTNAMES_GEORESOLVER),
        target_subnets=[get_prefix_from_ip(t) for t in pings_per_target.keys()],
        vp_subnets=[v["subnet"] for v in vps],
        target_ecs_table=ECS_TABLE,
        vps_ecs_table=ch_settings.VPS_ECS_MAPPING_TABLE,
    )

    vp_selection_per_target = get_vp_selection_per_target(
        output_path=RESULTS_PATH / "vp_selection.pickle",
        scores=scores,
        targets=pings_per_target.keys(),
        vps=vps,
    )

    meshed_shortest_pings = []
    georesolver_shortest_pings = []
    for target_addr, pings in pings_per_target.items():
        # get shortest ping using all vps
        meshed_shortest_ping = min(pings, key=lambda x: x[-1])
        meshed_shortest_pings.append(meshed_shortest_ping)

        # get shortest ping using georesolver VPs
        georesolver_vp_selection = vp_selection_per_target[target_addr][:50]
        georesolver_vp_selection = [v for v, _ in georesolver_vp_selection]
        georesolver_pings = []
        for vp_addr, vp_id, ping in pings:
            if vp_addr in georesolver_vp_selection:
                georesolver_pings.append((vp_addr, vp_id, ping))

        try:
            georesolver_shortest_ping = min(georesolver_pings, key=lambda x: x[-1])
            georesolver_shortest_pings.append(georesolver_shortest_ping)
        except:
            continue

    # plot latencies cdfs
    cdfs = []
    # all vps shortest pings
    x, y = ecdf([rtt for _, _, rtt in meshed_shortest_pings])
    cdfs.append([x, y, "Shortest ping, all VPs"])
    frac_under = round(get_proportion_under(x, y, 2), 2)
    logger.info(f"Meshed pings :: {frac_under=}")

    # georesolver shortesting ping
    x, y = ecdf([rtt for _, _, rtt in georesolver_shortest_pings])
    cdfs.append([x, y, "GeoResolver"])
    frac_under = round(get_proportion_under(x, y, 2), 2)
    logger.info(f"GeoResolver pings :: {frac_under=}")

    plot_multiple_cdf(
        cdfs=cdfs,
        output_path="cdns_latencies_comparison",
        metric_evaluated="rtt",
    )


def main() -> None:
    """
    entry point:
        - run meshed measurements per CDNs (all VPs towards one IP addr per /24 answer)
        - evaluate redirection latency
        - evaluate max distance based on exact position + geolocation area of presence
    """
    do_measurement: bool = False
    do_latency_eval: bool = False
    do_geo_eval: bool = False
    do_georesolver_comparison: bool = True

    prev_schedule_path: Path = (
        path_settings.MEASUREMENTS_SCHEDULE
        / "meshed_cdns_pings_test__2a0cc0c3-cb91-42bb-bf8b-20ef887390ed.json"
    )

    if do_measurement:
        asyncio.run(meshed_ping_cdns())
        # asyncio.run(meshed_ping_cdns(prev_schedule_path))
    if do_latency_eval:
        latency_eval()
    if do_geo_eval:
        geo_eval()
    if do_georesolver_comparison:
        cdn_meshed_vs_georesolver()


if __name__ == "__main__":
    main()
