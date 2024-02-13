import asyncio

from pyasn import pyasn
from numpy import mean
from tqdm import tqdm
from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import (
    OverallScore,
    GetDNSMappingHostnames,
    GetPoPInfo,
)

from geogiant.common.files_utils import dump_pickle, load_json, load_csv
from geogiant.common.ip_addresses_utils import get_prefix_from_ip, route_view_bgp_prefix
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def get_overall_score(
    target_subnet: list,
    column_name: str,
    hostname_filter: tuple[str],
) -> None:
    subnet_score = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await OverallScore().execute(
            client=client,
            table_name=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
            target_filter=target_subnet,
            column_name=column_name,
            hostname_filter=hostname_filter,
        )

    for row in resp:
        subnet_score[row["target_subnet"]] = row["scores"]

    return subnet_score


class VPSelectionECS:

    def __init__(
        self,
        targets: dict,
        vps: dict,
        hostname_filter: list,
        client_granularity: str = "client_bgp_prefix",
        answer_granularity: str = "answer_bgp_prefix",
    ) -> None:

        self.targets = targets
        self.vps = vps
        self.hostname_filter = hostname_filter
        self.client_granularity = client_granularity
        self.answer_granularity = answer_granularity

    async def get_subnets_mapping(self, subnets) -> dict:
        """get ecs-dns resolution per hostname for all input subnets"""
        async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
            # get vps mapping
            resp = await GetDNSMappingHostnames().execute_iter(
                client=client,
                table_name=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
                client_granularity=self.client_granularity,
                answer_granularity=self.answer_granularity,
                subnet_filter=[s for s in subnets],
                hostname_filter=self.hostname_filter,
            )

            subnets_mapping = defaultdict(dict)
            async for row in resp:
                subnet = row["client_granularity"]
                hostname = row["hostname"]
                mapping = row["mapping"]

                subnets_mapping[subnet][hostname] = mapping

        return subnets_mapping

    async def get_pop_info(self) -> dict:
        """get all geolocated PoP information"""
        async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:

            resp = await GetPoPInfo().execute_iter(
                client=client,
                table_name=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
                hostname_filter=self.hostname_filter,
            )

            pop_info = {}
            async for row in resp:
                pop = row[self.answer_granularity]
                del row[self.answer_granularity]
                pop_info[pop] = row

        return pop_info

    def get_target_granularity(self, target, asndb) -> str:
        """return the desired target/vp granularity (subnet or bgp prefix)"""
        match self.client_granularity:
            case "client_subnet":
                target_granularity = get_prefix_from_ip(target["address_v4"])
            case "client_bgp_prefix":
                asn, target_granularity = route_view_bgp_prefix(
                    target["address_v4"],
                    asndb,
                )
        return target_granularity

    def get_mapping_continent(self, mapping: list, pop_info: dict) -> set:
        """get the spanning continent for a given target/vp granularity (subnet or bgp prefix)"""
        pop_mapping_geoloc = set()
        for pop in mapping:
            pop_continent = pop_info[pop]["pop_continent"]
            # TODO: extend PoP geolocation
            if pop_continent != "Not found":
                pop_mapping_geoloc.add(pop_info[pop]["pop_continent"])

        return pop_mapping_geoloc

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

            target_mapping_continent = self.get_mapping_continent(
                target_mapping, pop_info
            )

            # if for a given hostname the target has a mapping covering two continent,
            # do not use hostname for geoloc for VP selection
            if len(target_mapping_continent) > 1:
                continue

            for vp_granularity in vps_mapping:
                try:
                    vp_mapping = vps_mapping[vp_granularity][hostname]
                except KeyError:
                    logger.error(
                        f"Hostname::{hostname} not found for vp {self.client_granularity} = {vp_granularity}"
                    )

                vp_mapping_continent = self.get_mapping_continent(vp_mapping, pop_info)

                if len(vp_mapping_continent) > 1:
                    continue

                # TODO: check for mapping continental disparity (distance matrix?)
                vps_score_per_hostname[vp_granularity].append(
                    len(set(target_mapping).intersection(set(vp_mapping)))
                    / min((len(set(target_mapping)), len(set(vp_mapping))))
                )

        return vps_score_per_hostname

    async def get_hostname_score(self) -> dict[list]:
        """for each target, compute the ecs fingerprint similarity for each VP"""

        asndb = pyasn(str(path_settings.RIB_TABLE))

        vps_mapping = await self.get_subnets_mapping(
            subnets=[get_prefix_from_ip(vp["address_v4"]) for vp in self.vps]
        )

        targets_mapping = await self.get_subnets_mapping(
            subnets=[get_prefix_from_ip(t["address_v4"]) for t in self.targets]
        )

        pop_info = await self.get_pop_info()

        # compute scores
        target_scores = {}
        for target in tqdm(self.targets):

            target_granularity = self.get_target_granularity(target, asndb)

            if not target_granularity:
                continue

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

        return target_scores


async def main() -> None:
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames_cdn.csv")

    scores = defaultdict(dict)

    logger.info("#############################################")
    logger.info("# HOSTNAME SCORE:: BGP PREFIXES             #")
    logger.info("#############################################")
    scores["client_bgp_prefix"]["answer_bgp_prefix"] = await VPSelectionECS(
        targets=targets,
        vps=vps,
        hostname_filter=hostname_filter,
        client_granularity="client_bgp_prefix",
        answer_granularity="answer_bgp_prefix",
    ).get_hostname_score()

    logger.info("#############################################")
    logger.info("# HOSTNAME SCORE:: SUBNETS                  #")
    logger.info("#############################################")
    scores["client_bgp_prefix"]["client_subnet"] = await VPSelectionECS(
        targets=targets,
        vps=vps,
        hostname_filter=hostname_filter,
        client_granularity="client_bgp_prefix",
        answer_granularity="answer_subnet",
    ).get_hostname_score()

    dump_pickle(
        data=scores, output_file=path_settings.RESULTS_PATH / "ecs_dns_scores.pickle"
    )

    # logger.info("#############################################")
    # logger.info("# OVERALL SCORE:: POP ID                    #")
    # logger.info("#############################################")
    # pop_id_score = await get_hostname_score(target_subnet, "pop_ip_info_id", hostname_filter)
    # dump_pickle(
    #     data=pop_id_score,
    #     output_file=path_settings.RESULTS_PATH / "pop_id_score.pickle",
    # )


if __name__ == "__main__":
    asyncio.run(main())
