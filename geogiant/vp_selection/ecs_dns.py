"""main functions for analyzing geolocation results"""

import asyncio

from dataclasses import dataclass
from collections import defaultdict
from loguru import logger
from tqdm import tqdm
from pych_client import AsyncClickHouseClient

from geogiant.vp_selection import VPSelectionBase
from geogiant.clickhouse import OverallScore

from geogiant.common.files_utils import load_json, load_csv, load_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


@dataclass(frozen=True)
class ResultsScore:
    client_granularity: str
    answer_granularity: str
    scores: list
    inconsistent_mappings: list


class VPSelectionDNS(VPSelectionBase):
    """use ECS-DNS resolution to select vps"""

    latency_threshold = 2

    async def get_overall_scores(self, targets: list, granularity: str) -> dict[list]:
        """for a list of targets return their dns mapping score"""
        scores = {}
        hostname_filter = load_csv(
            self.path_settings.DATASET / "valid_hostnames_cdn.csv"
        )

        target_subnet = list(
            set([get_prefix_from_ip(t["address_v4"]) for t in targets])
        )

        async with AsyncClickHouseClient(
            **self.clickhouse_settings.clickhouse
        ) as client:
            resp = await OverallScore().execute(
                client=client,
                table_name=self.clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
                target_filter=target_subnet,
                column_name=granularity,
                hostname_filter=hostname_filter,
            )

        for row in resp:
            scores[row["target_subnet"]] = row["scores"]

        return scores

    async def get_hostname_scores(self) -> dict[list]:
        """either calculate or simply return score file content"""
        scores: ResultsScore = load_pickle(
            path_settings.RESULTS_PATH
            / "score_not_extended_filtered_hostname_subnet.pickle"
        )
        return scores.scores

    def ecs_dns_vp_selection(
        self,
        vps_per_subnet: dict,
        ping_vps_to_target: dict,
        target_subnet_score: list,
        probing_budget: int = 50,
    ) -> None:
        """for a given target, select vps function of ecs-dns resolution"""

        highest_score_subnets = target_subnet_score[:probing_budget]

        vp_addrs, vp_coordinates = self.get_parsed_associated_vps(
            highest_score_subnets, vps_per_subnet
        )

        # get pings corresponding to the selected vps
        vp_selection = self.get_ping_to_target(
            target_associated_vps=vp_addrs,
            ping_to_target=ping_vps_to_target,
        )

        if not vp_selection:
            return None, probing_budget

        _, min_rtt = min(vp_selection, key=lambda x: x[-1])

        # filter out vps that are in the same city/AS
        vp_selection = self.select_one_vp_per_as_city(vp_selection, vp_coordinates)

        return vp_selection, probing_budget

    async def select_vps_per_target(
        self, probing_budget: int = 50, granularity: str = "answer_bgp_prefix"
    ) -> tuple[dict, set]:
        """return a sorted list of pair (vp,rtt) per target using ECS-DNS algo"""
        # load datasets
        targets = load_json(self.path_settings.OLD_TARGETS)
        vps = load_json(self.path_settings.OLD_VPS)

        vps_per_subnet = self.get_vps_per_subnet(vps)
        subnet_score = await self.get_hostname_scores()
        ping_vps_to_targets = await self.get_pings_per_target(
            self.clickhouse_settings.OLD_PING_VPS_TO_TARGET
        )

        logger.info(f"VP selection on {len(ping_vps_to_targets)} targets")

        target_unmapped = set()
        target_vp_selection = defaultdict(list)
        target_measurement_cost = defaultdict(int)
        for row in tqdm(ping_vps_to_targets):
            target_addr = row["target"]
            pings = row["pings"]

            target_subnet = get_prefix_from_ip(target_addr)

            (
                vp_selection,
                measurement_cost,
            ) = self.ecs_dns_vp_selection(
                vps_per_subnet=vps_per_subnet,
                ping_vps_to_target=pings,
                target_subnet_score=subnet_score[target_subnet],
                probing_budget=probing_budget,
            )

            if not vp_selection:
                target_unmapped.add(target_addr)
                continue

            target_vp_selection[target_addr] = vp_selection
            target_measurement_cost[target_addr] = measurement_cost

        return target_vp_selection, target_measurement_cost, target_unmapped

    async def select_vps_per_subnet(
        self, granularity: str = "answer_bgp_prefix"
    ) -> tuple[dict, set]:
        """return a sorted list of pair (vp, median_rtt) per subnet ECS-DNS algo"""
        # load datasets
        targets = load_json(self.path_settings.OLD_TARGETS)
        vps = load_json(self.path_settings.OLD_VPS)

        vps_per_subnet = self.get_vps_per_subnet(vps)
        ping_vps_to_subnet = await self.get_pings_per_subnet(
            self.clickhouse_settings.OLD_PING_VPS_TO_SUBNET
        )
        subnet_score = await self.get_hostname_scores()

        logger.info(f"VP selection on {len(ping_vps_to_subnet)} subnets")

        subnet_unmapped = set()
        subnet_vp_selection = defaultdict(list)
        subnet_measurement_cost = defaultdict(int)
        for subnet in ping_vps_to_subnet:
            min_rtts_per_vp = defaultdict(list)

            # get all vps rtt for the three representative, so we can calculate median error
            for target_addr, ping_vps_to_target in ping_vps_to_subnet[subnet].items():
                target_subnet = get_prefix_from_ip(target_addr)

                (
                    vp_selection,
                    measurement_cost,
                ) = self.ecs_dns_vp_selection(
                    vps_per_subnet=vps_per_subnet,
                    ping_vps_to_target=ping_vps_to_target,
                    target_subnet_score=subnet_score[target_subnet],
                )

                subnet_measurement_cost[subnet] += measurement_cost

                if not vp_selection:
                    continue

                # get rtts to each representative for each subnet target
                for vp_addr, min_rtt in vp_selection:
                    min_rtts_per_vp[vp_addr].append(min_rtt)

            # Calculate median errors to the three representative and order VPs
            subnet_vp_selection[subnet] = self.get_best_subnet_vp(min_rtts_per_vp)

        return subnet_vp_selection, subnet_measurement_cost, subnet_unmapped


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(
        VPSelectionDNS().main(
            targets=targets,
            vps=vps,
            output_path="ecs_dns",
            target_selection=True,
            subnet_selection=True,
        )
    )
