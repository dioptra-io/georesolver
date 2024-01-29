"""main functions for analyzing geolocation results"""
import asyncio

from collections import defaultdict
from loguru import logger
from tqdm import tqdm
from pych_client import AsyncClickHouseClient

from geogiant.vp_selection import VPSelectionBase
from geogiant.clickhouse import (
    OverallScore,
    OverallPoPSubnetScore,
    HostnamePoPFrontendScore,
    HostnamePoPSubnetScore,
)

from geogiant.common.files_utils import load_json, load_pickle
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


hostname_filter = (
    "outlook.live.com",
    "docs.edgecast.com",
    "advancedhosting.com",
    "tencentcloud.com",
    "teams.microsoft.com",
    "cachefly.com",
    "chrome.google.com",
    "calendar.google.com",
    "business.google.com",
    "classroom.google.com",
    "www.youtube.com",
    "accounts.google.com",
    "docs.google.com",
    "drive.google.com",
    "mail.google.com",
    "meet.google.com",
    "news.google.com",
    "one.google.com",
    "photos.google.com",
    "scholar.google.com",
    "sites.google.com",
    "studio.youtube.com",
    "support.google.com",
    "www.google.ca",
    "www.google.cl",
    "www.google.co.in",
    "www.google.co.jp",
    "www.google.com.ar",
    "www.google.co.uk",
    "www.google.com.tw",
    "www.google.de",
    "www.google.fr",
    "www.google.it",
    "www.google.nl",
    "www.google.p",
    "www.google.co.th",
    "www.google.pl",
    "www.google.com.tr",
    "myactivity.google.com",
    "translate.google.com",
    "myaccount.google.com",
    "play.google.com",
    "www.google.com.br",
    "www.google.es",
    "www.google.com.mx",
    "www.google.com.hk",
    "www.yahoo.co.jp",
    "weather.yahoo.co.jp",
    "search.yahoo.co.jp",
    "page.auctions.yahoo.co.jp",
    "detail.chiebukuro.yahoo.co.jp",
    "baseball.yahoo.co.jp",
    "auctions.yahoo.co.jp",
    "apps.facebook.com",
)


class VPSelectionDNS(VPSelectionBase):
    """use ECS-DNS resolution to select vps"""

    latency_threshold = 1

    async def get_subnet_score(self, targets: list) -> dict[list]:
        """for a list of targets return their dns mapping score"""
        subnet_score = {}

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
                column_name="answer_bgp_prefix",
                hostname_filter=hostname_filter,
            )

        for row in resp:
            subnet_score[row["target_subnet"]] = row["scores"]

        return subnet_score

    def ecs_dns_vp_selection(
        self,
        vps_per_subnet: dict,
        ping_vps_to_target: dict,
        target_subnet_score: list,
        max_probing_budget: int = 50,
    ) -> None:
        """for a given target, select vps function of ecs-dns resolution"""

        for probing_budget in range(10, max_probing_budget + 10, 10):
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
                continue

            _, min_rtt = min(vp_selection, key=lambda x: x[-1])

            # check if dns vp selection sufficient
            if min_rtt < self.latency_threshold:
                break

        # filter out vps that are in the same city/AS
        vp_selection = self.select_one_vp_per_as_city(vp_selection, vp_coordinates)

        return vp_selection, probing_budget

    async def select_vps_per_target(self) -> [dict, set]:
        """return a sorted list of pair (vp,rtt) per target using ECS-DNS algo"""
        # load datasets
        targets = load_json(self.path_settings.OLD_TARGETS)
        vps = load_json(self.path_settings.OLD_VPS)
        vps_per_subnet = self.get_vps_per_subnet(vps)
        subnet_score = await self.get_subnet_score(targets=targets)
        ping_vps_to_targets = await self.get_pings_per_target(
            self.clickhouse_settings.OLD_PING_VPS_TO_TARGET
        )

        logger.info(f"VP selection on {len(ping_vps_to_targets)} targets")

        target_vp_selection = defaultdict(list)
        target_measurement_cost = defaultdict(int)
        target_unmapped = set()

        for row in tqdm(ping_vps_to_targets):
            target_addr = row["target"]
            pings = row["pings"]

            target_subnet = get_prefix_from_ip(target_addr)

            (
                target_vp_selection[target_addr],
                target_measurement_cost[target_addr],
            ) = self.ecs_dns_vp_selection(
                vps_per_subnet=vps_per_subnet,
                ping_vps_to_target=pings,
                target_subnet_score=subnet_score[target_subnet],
            )

            if not target_vp_selection[target_addr]:
                target_unmapped.add(target_addr)

        return target_vp_selection, target_measurement_cost, target_unmapped

    async def select_vps_per_subnet(self) -> [dict, set]:
        """return a sorted list of pair (vp, median_rtt) per subnet ECS-DNS algo"""
        # load datasets
        targets = load_json(self.path_settings.OLD_TARGETS)
        vps = load_json(self.path_settings.OLD_VPS)
        vps_per_subnet = self.get_vps_per_subnet(vps)
        subnet_score = await self.get_subnet_score(targets=targets)
        ping_vps_to_subnet = await self.get_avg_rtt_per_subnet(
            self.clickhouse_settings.OLD_PING_VPS_TO_SUBNET
        )

        logger.info(f"VP selection on {len(ping_vps_to_subnet)} subnets")

        subnet_vp_selection = defaultdict(list)
        subnet_measurement_cost = defaultdict(int)
        subnet_unmapped = set()

        for row in ping_vps_to_subnet:
            subnet = row["subnet"]
            vp = row["vp"]
            vps_avg_rtt = row["vps_avg_rtt"]

            min_rtts_per_vp = defaultdict(list)
            # get all vps rtt for the three representative, so we can calculate median error
            for target_addr, ping_vps_to_target in ping_vps_to_subnet[subnet].items():
                target_subnet = get_prefix_from_ip(target_addr)

                (
                    vp_selection,
                    measurement_cost,
                ) = self.ecs_dns_vp_selection(
                    vps_per_subnet=vps_per_subnet,
                    ping_vps_to_target=ping_vps_to_subnet[target_subnet],
                    target_subnet_score=subnet_score[target_subnet],
                )

                subnet_measurement_cost[subnet] += measurement_cost

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
            subnet_selection=False,
        )
    )
