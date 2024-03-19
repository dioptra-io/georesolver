import asyncio
import random

from pyasn import pyasn
from numpy import mean
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from geogiant.ecs_vp_selection.query import (
    get_subnets_mapping,
    get_pop_info,
    get_pop_per_hostname,
    get_subnet_per_pop,
    load_targets,
    load_vps,
)
from geogiant.ecs_vp_selection.utils import (
    ResultsScore,
)
from geogiant.common.geoloc import distance
from geogiant.common.files_utils import dump_pickle, load_json, load_csv
from geogiant.common.ip_addresses_utils import (
    get_prefix_from_ip,
    get_addr_granularity,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


class VPSelectionECS:

    def __init__(
        self,
        targets: dict,
        vps: dict,
        dns_table: str,
        hostname_filter: list,
        client_granularity: str = "client_bgp_prefix",
        answer_granularity: str = "answer_bgp_prefix",
        check_mapping_consistency: bool = False,
    ) -> None:

        self.targets = targets
        self.vps = vps
        self.dns_table = dns_table
        self.hostname_filter = hostname_filter
        self.client_granularity = client_granularity
        self.answer_granularity = answer_granularity
        self.check_mapping_consistency = check_mapping_consistency

    def get_mapping_continent(self, mapping: list, pop_info: dict) -> set:
        """get the spanning continent for a given target/vp granularity (subnet or bgp prefix)"""
        pop_mapping_geoloc = set()
        for pop in mapping:
            pop_continent = pop_info[pop]["pop_continent"]
            # TODO: extend PoP geolocation
            if pop_continent != "Not found":
                pop_mapping_geoloc.add(pop_continent)

        return pop_mapping_geoloc

    def get_major_geo_ratio(self, mapping: list, pop_info: dict) -> float:
        """compute major continent ratio for a given target ecs fingerprint"""
        pop_mapping_geoloc = []
        for pop in mapping:
            pop_continent = pop_info[pop]["pop_continent"]
            if pop_continent != "Not found":
                pop_mapping_geoloc.append(pop_continent)

        # get most represented region ratio
        if pop_mapping_geoloc:
            return max(
                [
                    pop_mapping_geoloc.count(region) / len(pop_mapping_geoloc)
                    for region in pop_mapping_geoloc
                ]
            )

        return 0

    def get_target_mapping_geoloc_ratio(
        self, target_mapping: dict, pop_info: dict
    ) -> None:
        """check if a major geographic region can be found for a given target"""
        target_overall_mapping = []
        for target_mapping in target_mapping.values():
            target_overall_mapping.extend([pop_subnet for pop_subnet in target_mapping])

        return self.get_major_geo_ratio(target_overall_mapping, pop_info)

    def get_vp_to_pops_dst(
        self, vp_lat: float, vp_lon: float, hostname_pops: dict
    ) -> list:
        """calculate distance between VPs and PoPs"""
        vp_to_pops_dst = []
        for pop_subnet, (pop_lat, pop_lon) in hostname_pops.items():
            if pop_lat != -1 and pop_lon != -1:
                d = distance(vp_lat, pop_lat, vp_lon, pop_lon)
                vp_to_pops_dst.append((pop_subnet, pop_lat, pop_lon, d))

        return vp_to_pops_dst

    async def get_extended_vp_mapping(
        self,
        vps_in_subnet: list[dict],
        vp_mapping: dict[str, list],
        pop_per_hostname: dict,
        subnet_per_pop: dict,
    ) -> None:
        """for each vps subnet, get the associated best PoP, based on PoP geolocation"""
        extended_vp_mapping = defaultdict(dict)

        for vp in vps_in_subnet:
            (vp_lon, vp_lat) = vp["geometry"]["coordinates"]

            # find the best PoP for each hostname
            for hostname, mapping in vp_mapping.items():
                hostname_pops: dict = pop_per_hostname[hostname]

                vp_to_pops_dst = self.get_vp_to_pops_dst(vp_lat, vp_lon, hostname_pops)

                # extend mapping with closest PoP subnets
                if vp_to_pops_dst:
                    closest_pop_subnets = [
                        pop_subnet for pop_subnet, _, _, d in vp_to_pops_dst if d < 100
                    ]

                    if not closest_pop_subnets:
                        closest_pop = min(vp_to_pops_dst, key=lambda x: x[-1])
                        closest_pop_lat, closest_pop_lon = (
                            closest_pop[1],
                            closest_pop[2],
                        )
                        closest_pop_subnets = subnet_per_pop[hostname][
                            (closest_pop_lat, closest_pop_lon)
                        ]

                    mapping.extend(closest_pop_subnets)

                extended_vp_mapping[hostname] = list(set(mapping))

        return extended_vp_mapping

    def get_vps_score_per_hostname(
        self,
        target_granularity: str,
        targets_mapping: dict[dict],
        vps_mapping: dict,
        pop_info: dict,
    ) -> dict[list]:
        """get vps score per hostname for a given target"""
        vps_score_per_hostname = defaultdict(list)

        for hostname, target_mapping in targets_mapping[target_granularity].items():

            if self.check_mapping_consistency:
                # if for a given hostname the target has a mapping covering two continent,
                # do not use hostname for geoloc for VP selection
                target_mapping_continent = self.get_mapping_continent(
                    target_mapping, pop_info
                )
                if len(target_mapping_continent) > 1:
                    pass

            # compute VP similarity for hostname
            for vp_granularity in vps_mapping:
                try:
                    vp_mapping = vps_mapping[vp_granularity][hostname]
                except KeyError:
                    continue

                if self.check_mapping_consistency:
                    vp_mapping_continent = self.get_mapping_continent(
                        vp_mapping, pop_info
                    )
                    if len(vp_mapping_continent) > 1:
                        continue

                vps_score_per_hostname[vp_granularity].append(
                    len(set(target_mapping).intersection(set(vp_mapping)))
                    / min((len(set(target_mapping)), len(set(vp_mapping))))
                )

        return vps_score_per_hostname

    async def get_vps_mapping_with_closest_pop(self, vps_mapping: dict) -> dict:
        """for a given resolution append PoP subnets that are close to a given vp subnet"""
        extended_vps_mapping = {}
        pop_per_hostname = await get_pop_per_hostname(
            dns_table=self.dns_table, hostname_filter=self.hostname_filter
        )
        subnet_per_pop = await get_subnet_per_pop(
            dns_table=self.dns_table, hostname_filter=self.hostname_filter
        )

        vps_per_subnet = defaultdict(list)
        for vp in self.vps:
            vp_addr = vp["address_v4"]
            vp_subnet = get_prefix_from_ip(vp_addr)
            vps_per_subnet[vp_subnet].append(vp)

        for vp_subnet, vp_mapping in vps_mapping.items():
            extended_vps_mapping[vp_subnet] = await self.get_extended_vp_mapping(
                vps_in_subnet=vps_per_subnet[vp_subnet],
                vp_mapping=vp_mapping,
                pop_per_hostname=pop_per_hostname,
                subnet_per_pop=subnet_per_pop,
            )

        return extended_vps_mapping

    async def get_hostname_score(
        self,
        extend_vp_mapping: bool = False,
        inconsistent_mapping_threshold: float = 0.5,
    ) -> dict[list]:
        """for each target, compute the ecs fingerprint similarity for each VP"""
        asndb = pyasn(str(path_settings.RIB_TABLE))

        targets_mapping = await get_subnets_mapping(
            dns_table=self.dns_table,
            answer_granularity=self.answer_granularity,
            subnets=[get_prefix_from_ip(t["address_v4"]) for t in self.targets],
            hostname_filter=self.hostname_filter,
        )
        vps_mapping = await get_subnets_mapping(
            dns_table=self.dns_table,
            answer_granularity=self.answer_granularity,
            subnets=[get_prefix_from_ip(vp["address_v4"]) for vp in self.vps],
            hostname_filter=self.hostname_filter,
        )

        pop_info = None
        if self.check_mapping_consistency:
            pop_info = await get_pop_info(
                dns_table=self.dns_table,
                answer_granularity=self.answer_granularity,
                hostname_filter=self.hostname_filter,
            )

        # add PoP subnets under 100 [km] from VP
        if extend_vp_mapping:
            logger.info(f"VP selection with extended mapping")
            vps_mapping = await self.get_vps_mapping_with_closest_pop(vps_mapping)

        target_scores = {}
        inconsistent_mappings = []
        for target in tqdm(self.targets):

            target_granularity = get_addr_granularity(
                self.client_granularity, target, asndb
            )

            # TODO: complete pyasn when bgp_prefix not found, ok for target subnet
            if not target_granularity:
                logger.error(f"BGP prefix not found for target: {target['address_v4']}")
                continue

            if self.check_mapping_consistency:
                geoloc_ratio = self.get_target_mapping_geoloc_ratio(
                    target_mapping=targets_mapping[target_granularity],
                    pop_info=pop_info,
                )

                if geoloc_ratio < inconsistent_mapping_threshold:
                    logger.info(
                        f"Target::{target_granularity} mapping inconsistent, major country ratio = {geoloc_ratio}"
                    )

            # check for mapping inconsistency (i.e. no major geographic region)
            vps_score_per_hostname = self.get_vps_score_per_hostname(
                target_granularity=target_granularity,
                targets_mapping=targets_mapping,
                vps_mapping=vps_mapping,
                pop_info=pop_info,
            )

            # get the avg number of match across hostnames as mapping score
            target_scores[target_granularity] = sorted(
                [
                    (vp_subnet, mean(hostname_scores))
                    for vp_subnet, hostname_scores in vps_score_per_hostname.items()
                ],
                key=lambda x: x[-1],
                reverse=True,
            )

        return ResultsScore(
            client_granularity=self.client_granularity,
            answer_granularity=self.answer_granularity,
            scores=target_scores,
            inconsistent_mappings=inconsistent_mappings,
        )


async def main() -> None:
    score_answer = False
    score_subnet = False
    score_bgp = False
    score_pop = False
    score_per_cdn = False
    score_all_hostnames = True
    score_all_hostnames_per_cnd = False

    extend_vp_mapping = False
    filtered_hostname = True

    base_output_file = f"score_{'extended_mapping' if extend_vp_mapping else 'not_extended'}_{'filtered_hostname' if filtered_hostname else 'no_filtered'}"

    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    if filtered_hostname:
        hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames_cdn.csv")
    else:
        hostname_filter = ""

    if score_answer:
        output_file = path_settings.RESULTS_PATH / (
            base_output_file + f"_answer.pickle"
        )
        logger.info("#############################################")
        logger.info("# HOSTNAME SCORE:: ANSWERS                  #")
        logger.info("#############################################")
        logger.info(f"Results at: {output_file}")
        score_answer = await VPSelectionECS(
            targets=targets,
            vps=vps,
            hostname_filter=hostname_filter,
            dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
            client_granularity="client_subnet",
            answer_granularity="answer",
        ).get_hostname_score(
            extend_vp_mapping=extend_vp_mapping,
            inconsistent_mapping_threshold=0.5,
        )

        dump_pickle(data=score_answer, output_file=output_file)

    if score_subnet:
        output_file = path_settings.RESULTS_PATH / (
            base_output_file + f"_subnet.pickle"
        )
        logger.info("#############################################")
        logger.info("# HOSTNAME SCORE:: SUBNETS                  #")
        logger.info("#############################################")
        logger.info(f"Results at: {output_file}")
        score_subnet = await VPSelectionECS(
            targets=targets,
            vps=vps,
            hostname_filter=hostname_filter,
            dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
            client_granularity="client_subnet",
            answer_granularity="answer_subnet",
        ).get_hostname_score(
            extend_vp_mapping=extend_vp_mapping,
            inconsistent_mapping_threshold=0.5,
        )

        dump_pickle(data=score_subnet, output_file=output_file)

    if score_bgp:
        logger.info("#############################################")
        logger.info("# HOSTNAME SCORE:: BGP PREFIXES             #")
        logger.info("#############################################")
        output_file = path_settings.RESULTS_PATH / (
            base_output_file + f"_bgp_prefix.pickle"
        )
        score_bgp = await VPSelectionECS(
            targets=targets,
            vps=vps,
            hostname_filter=hostname_filter,
            dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
            client_granularity="client_subnet",
            answer_granularity="answer_bgp_prefix",
        ).get_hostname_score(
            extend_vp_mapping=extend_vp_mapping,
            inconsistent_mapping_threshold=0.5,
        )

        dump_pickle(data=score_bgp, output_file=output_file)

    if score_pop:
        logger.info("#############################################")
        logger.info("# HOSTNAME SCORE:: POP CITY                 #")
        logger.info("#############################################")
        output_file = path_settings.RESULTS_PATH / (base_output_file + f"_pop.pickle")
        score_pop = await VPSelectionECS(
            targets=targets,
            vps=vps,
            hostname_filter=hostname_filter,
            client_granularity="client_subnet",
            answer_granularity="pop_city",
        ).get_hostname_score()

        dump_pickle(data=score_pop, output_file=output_file)

    if score_per_cdn:
        hostname_per_cdns = {
            "facebook": ["apps.facebook.com"],
            "google": ["www.google.com"],
            "amazon": ["aws.amazon.com"],
            "cdn_networks": ["www.cdnetworks.com"],
            "apple": ["swcdn.apple.com"],
        }

        score_per_cdn = {}
        for cdn, hostnames in hostname_per_cdns.items():
            logger.info(f"Compute scores for:: {cdn=}")
            output_file = path_settings.RESULTS_PATH / (
                base_output_file + f"_{cdn}.pickle"
            )
            score_per_cdn = await VPSelectionECS(
                targets=targets,
                vps=vps,
                dns_table=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
                hostname_filter=hostnames,
                client_granularity="client_subnet",
                answer_granularity="answer_subnet",
            ).get_hostname_score()

            dump_pickle(data=score_per_cdn, output_file=output_file)

    if score_all_hostnames:
        greedy_selection = False
        greedy_bgp = False
        greedy_cdn = False
        max_bgp_prefix_per_cdn = True
        answer_granularity = "answer_bgp_prefix"
        all_hostname_file = base_output_file + f"_1M_hostnames_greedy_cdn"

        targets = await load_targets(clickhouse_settings.VPS_RAW)
        vps = await load_vps(clickhouse_settings.VPS_RAW)

        if greedy_selection:
            if greedy_cdn:
                hostname_filter = load_csv(
                    path_settings.DATASET / "hostname_1M_greedy_cdn.csv"
                )
            if greedy_bgp:
                hostname_filter = load_csv(
                    path_settings.DATASET / "hostname_1M_greedy_bgp.csv"
                )

        if max_bgp_prefix_per_cdn:
            hostname_filter = load_csv(
                path_settings.DATASET / "hostname_1M_max_bgp_prefix_per_cdn.csv"
            )
        else:
            hostname_filter = load_csv(path_settings.DATASET / "elected_hostnames.csv")

        hostname_filter = [row.split(",")[0] for row in hostname_filter]

        logger.info("#############################################")
        logger.info("# HOSTNAME SCORE:: ALL HOSTNAMES            #")
        logger.info("#############################################")
        output_file = path_settings.RESULTS_PATH / (
            all_hostname_file + f"_{answer_granularity}_new.pickle"
        )
        score_all_hostnames = await VPSelectionECS(
            targets=targets,
            vps=vps,
            dns_table="filtered_hostnames_ecs_mapping",
            hostname_filter=hostname_filter,
            client_granularity="subnet",
            answer_granularity=answer_granularity,
        ).get_hostname_score()

        dump_pickle(data=score_all_hostnames, output_file=output_file)

    if score_all_hostnames_per_cnd:
        answer_granularity = "answer_bgp_prefix"

        targets = await load_targets(clickhouse_settings.VPS_RAW)
        vps = await load_vps(clickhouse_settings.VPS_RAW)

        hostname_per_cdn = load_json(path_settings.DATASET / "hostname_per_cdn_1M.json")

        for cdn, hostnames in hostname_per_cdn.items():

            logger.info(f"Computing ECS scores for:: {cdn=}, {hostnames=}")

            output_file = path_settings.RESULTS_PATH / (
                base_output_file + f"_1M_hostnames_{answer_granularity}_{cdn}.pickle"
            )

            score_cdn = await VPSelectionECS(
                targets=targets,
                vps=vps,
                dns_table=clickhouse_settings.DNS_MAPPING_VPS_RAW,
                hostname_filter=hostnames,
                client_granularity="subnet",
                answer_granularity=answer_granularity,
            ).get_hostname_score()

            dump_pickle(data=score_cdn, output_file=output_file)


if __name__ == "__main__":
    asyncio.run(main())
