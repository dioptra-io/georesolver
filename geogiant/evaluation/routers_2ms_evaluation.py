import asyncio
import json

from pyasn import pyasn
from ipaddress import IPv4Address, AddressValueError
from typing import Generator
from pathlib import Path
from numpy import mean
from tqdm import tqdm
from loguru import logger
from collections import defaultdict
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import GetSubnets, GetVPsAndAnchors

from geogiant.ecs_vp_selection.query import get_subnets_mapping
from geogiant.ecs_vp_selection.utils import Score, get_parsed_vps
from geogiant.common.geoloc import distance
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.files_utils import (
    dump_pickle,
    load_csv,
    load_pickle,
    dump_json,
    load_json,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


class VPSelectionECS:
    def __init__(
        self,
        hostname_filter: list,
        answer_granularity: str = "answer_bgp_prefix",
    ) -> None:

        self.hostname_filter = hostname_filter
        self.answer_granularity = answer_granularity

    def get_vps_score_per_hostname(
        self,
        target_subnet: str,
        targets_mapping: dict[dict],
        vps_mapping: dict,
    ) -> dict[list]:
        """get vps score per hostname for a given target"""
        vps_score_per_hostname = defaultdict(list)

        for hostname, target_mapping in targets_mapping[target_subnet].items():

            # compute VP similarity for hostname
            for vp_subnet in vps_mapping:
                try:
                    vp_mapping = vps_mapping[vp_subnet][hostname]
                except KeyError:
                    continue

                vps_score_per_hostname[vp_subnet].append(
                    len(set(target_mapping).intersection(set(vp_mapping)))
                    / min((len(set(target_mapping)), len(set(vp_mapping))))
                )

        return vps_score_per_hostname

    async def get_hostname_score(
        self,
        targets_subnet: list,
        vps_subnet: list,
    ) -> dict[list]:
        """for each target, compute the ecs fingerprint similarity for each VP"""
        targets_mapping = await get_subnets_mapping(
            dns_table="dns_mapping_routers_2ms",
            answer_granularity=self.answer_granularity,
            subnets=[subnet for subnet in targets_subnet],
            hostname_filter=self.hostname_filter,
        )
        vps_mapping = await get_subnets_mapping(
            dns_table="filtered_hostnames_ecs_mapping",
            answer_granularity=self.answer_granularity,
            subnets=[subnet for subnet in vps_subnet],
            hostname_filter=self.hostname_filter,
        )

        target_scores = {}
        for target_subnet in tqdm(targets_subnet):

            # check for mapping inconsistency (i.e. no major geographic region)
            vps_score_per_hostname = self.get_vps_score_per_hostname(
                target_subnet=target_subnet,
                targets_mapping=targets_mapping,
                vps_mapping=vps_mapping,
            )

            # get the avg number of match across hostnames as mapping score
            target_scores[target_subnet] = sorted(
                [
                    (vp_subnet, mean(hostname_scores))
                    for vp_subnet, hostname_scores in vps_score_per_hostname.items()
                ],
                key=lambda x: x[-1],
                reverse=True,
            )

            target_scores[target_subnet] = target_scores[target_subnet][:1_000]

        return Score(
            answer_granularity=self.answer_granularity,
            scores=target_scores,
        )


def load_json_iter(file_path: Path) -> Generator:
    """iter load json file"""
    with file_path.open("r") as f:
        for row in f.readlines():
            yield json.loads(row)


async def compute_scores() -> None:
    # answer_granularity = "answer_bgp_prefix"
    answer_granularity = "answer_subnet"
    target_table = "dns_mapping_routers_2ms"
    vps_table = "filtered_hostnames_ecs_mapping"
    output_file = (
        path_settings.RESULTS_PATH
        / f"score_1M_hostnames_routers_2ms_{answer_granularity}.pickle"
    )

    # TODO: get targets
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = await GetSubnets().execute(client=client, table_name=target_table)
        target_subnets = [row["subnet"] for row in rows]

        rows = await GetSubnets().execute(client=client, table_name=vps_table)
        vps_subnets = [row["subnet"] for row in rows]

    hostname_filter = load_csv(
        path_settings.DATASET / "hostname_1M_max_bgp_prefix_per_cdn.csv"
    )
    hostname_filter = [row.split(",")[0] for row in hostname_filter]

    logger.info("#############################################")
    logger.info("# HOSTNAME SCORE:: ROUTERS 2ms              #")
    logger.info("#############################################")
    logger.info(f"Scores output file: {output_file}")
    score_all_hostnames = await VPSelectionECS(
        hostname_filter=hostname_filter,
        answer_granularity=answer_granularity,
    ).get_hostname_score(target_subnets, vps_subnets)

    dump_pickle(data=score_all_hostnames, output_file=output_file)


def get_vp_info(
    target: dict,
    target_score: list,
    vp_addr: str,
    vps_coordinates: dict,
    rtt: float = None,
) -> dict:
    """get all useful information about a selected VP"""
    vp_subnet = get_prefix_from_ip(vp_addr)
    lat, lon, _ = vps_coordinates[vp_addr]
    d_error = distance(target["lat"], lat, target["lon"], lon)

    return {
        "addr": vp_addr,
        "subnet": vp_subnet,
        "lat": lat,
        "lon": lon,
        "rtt": rtt,
        "d_error": d_error,
    }


def get_no_ping_vp(
    target, target_score: list, vps_per_subnet: dict, vps_coordinates: dict
) -> dict:
    """return VP with maximum score"""
    subnet, _ = target_score[0]
    vp_addr = vps_per_subnet[subnet][0]

    return get_vp_info(target, target_score, vp_addr, vps_coordinates)


async def load_vps(dns_table: str) -> dict:
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        vps = await GetVPsAndAnchors().execute(
            client=client,
            table_name=dns_table,
        )

        return vps


def get_parsed_vps(vps: list, asndb: pyasn) -> dict:
    """parse vps list to a dict for fast retrieval. Keys depends on granularity"""
    vps_coordinates = {}
    vps_subnet = defaultdict(list)
    vps_bgp_prefix = defaultdict(list)

    for vp in vps:
        vp_addr = vp["address_v4"]
        subnet = get_prefix_from_ip(vp_addr)
        vp_asn, vp_bgp_prefix = route_view_bgp_prefix(vp_addr, asndb)
        vp_lat, vp_lon = vp["lat"], vp["lon"]

        vps_subnet[subnet].append(vp_addr)
        vps_bgp_prefix[vp_bgp_prefix].append(vp_addr)
        vps_coordinates[vp_addr] = (vp_lat, vp_lon, vp_asn)

    return vps_subnet, vps_bgp_prefix, vps_coordinates


async def geoloc_error() -> None:
    """compute geoloc error over all targets"""
    asndb = pyasn(str(path_settings.RIB_TABLE))
    score_output_file = (
        path_settings.RESULTS_PATH
        / f"score_1M_hostnames_routers_2ms_answer_bgp_prefix.pickle"
    )
    target_path = path_settings.DATASET / "router_2ms_targets.json"

    ecs_scores = load_pickle(score_output_file)
    ecs_scores = ecs_scores.scores
    rows = load_json_iter(path_settings.DATASET / "routers_2ms.json")
    vps = await load_vps(clickhouse_settings.VPS_RAW)

    vps_per_subnet, vps_bgp_prefix, vps_coordinates = get_parsed_vps(vps, asndb)

    if not target_path.exists():
        targets = {}
        for row in rows:
            addr = row["ip"]
            try:
                if not IPv4Address(addr).is_private:
                    subnet = get_prefix_from_ip(addr)
                    try:
                        _ = ecs_scores[subnet]
                        target = {
                            "addr": addr,
                            "subnet": subnet,
                            "lat": row["probe_latitude"],
                            "lon": row["probe_longitude"],
                        }
                        targets[addr] = target
                    except KeyError:
                        continue
            except AddressValueError:
                continue

        dump_json(data=targets, output_file=target_path)

    targets = load_json(target_path)

    results = {}
    logger.info(f"Analysis on :: {len(targets)} targets")
    for target, target_info in tqdm(targets.items()):
        target_subnet = str(target_info["subnet"])
        target_subnet = get_prefix_from_ip(target)
        if not ecs_scores[target_subnet]:
            logger.error(f"No score for target:: {target_subnet}")
            continue

        no_ping_vp_subnet, _ = ecs_scores[target_subnet][0]
        no_ping_vp_addr = vps_per_subnet[no_ping_vp_subnet][0]

        ecs_vps = ecs_scores[target_subnet][:50]
        ecs_vps_10 = ecs_scores[target_subnet][:10]

        all_vps_distance = []
        for vp, (vp_lat, vp_lon, _) in vps_coordinates.items():
            d = distance(target_info["lat"], vp_lat, target_info["lon"], vp_lon)
            all_vps_distance.append((vp, d))

        ref_addr, _ = min(all_vps_distance, key=lambda x: x[-1])
        ref_vp = get_vp_info(
            target=target_info,
            target_score=ecs_scores[target_subnet][0],
            vp_addr=ref_addr,
            vps_coordinates=vps_coordinates,
        )

        elected_vps = []
        for vp_subnet, score in ecs_vps:
            vp_addr = vps_per_subnet[vp_subnet][0]
            vp_lat, vp_lon, _ = vps_coordinates[vp_addr]
            d = distance(target_info["lat"], vp_lat, target_info["lon"], vp_lon)
            elected_vps.append((vp_addr, d))

        # elected_vps_10 = []
        # for vp_subnet, score in ecs_vps_10:
        #     vp_addr = vps_per_subnet[vp_subnet][0]
        #     vp_lat, vp_lon, _ = vps_coordinates[vp_addr]
        #     d = distance(target_info["lat"], vp_lat, target_info["lon"], vp_lon)
        #     elected_vps_10.append((vp_addr, d))

        closest_ecs_vp_addr, _ = min(elected_vps, key=lambda x: x[1])
        closest_ecs_vp = get_vp_info(
            target=target_info,
            target_score=ecs_scores[target_subnet][0],
            vp_addr=closest_ecs_vp_addr,
            vps_coordinates=vps_coordinates,
        )

        # closest_ecs_vp_addr_10, _ = min(elected_vps_10, key=lambda x: x[1])
        # closest_ecs_vp_10 = get_vp_info(
        #     target=target_info,
        #     target_score=ecs_scores[target_subnet][0],
        #     vp_addr=closest_ecs_vp_addr_10,
        #     vps_coordinates=vps_coordinates,
        # )

        no_ping_vp = get_vp_info(
            target=target_info,
            target_score=ecs_scores[target_subnet][0],
            vp_addr=no_ping_vp_addr,
            vps_coordinates=vps_coordinates,
        )

        results[target] = {
            "target": target_info,
            "ref_shortest_ping_vp": ref_vp,
            "ecs_shortest_ping_vp": closest_ecs_vp,
            "no_ping_vp": no_ping_vp,
        }

    dump_pickle(
        data=results,
        output_file=path_settings.RESULTS_PATH / f"evaluation_routers_2ms.pickle",
    )


async def main() -> None:
    # await compute_scores()
    await geoloc_error()


if __name__ == "__main__":
    asyncio.run(main())
