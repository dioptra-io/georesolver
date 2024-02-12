"""main functions for analyzing geolocation results"""

import asyncio
from collections import defaultdict

from loguru import logger
from tqdm import tqdm

from geogiant.common.files_utils import load_json
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings
from geogiant.common.geoloc import cbg, rtt_to_km, distance
from geogiant.vp_selection import VPSelectionDNS

path_settings = PathSettings()


class VPSelectionMultiIteration(VPSelectionDNS):
    """mixed vp selection, first use ECS-DNS resolution for selection
    then use geographical location around vp with the lowest latency
    """

    latency_threshold: int = 1
    nb_round: int = 1
    probing_budget: int = 30

    def get_vp_within_geographical_area(
        self,
        vps_coordinates: dict,
        center_coordinates: tuple,
        radius: float,
        threshold: int = 20,
    ) -> list:
        """From a pop geolocation, return all vps within this area"""
        target_associated_vps = []
        selected_vps_per_asn = defaultdict(list)

        center_lat, center_lon = center_coordinates

        if not center_lat or not center_lat:
            return []

        # for each vp calculate if within geographic area
        for vp_addr, (vp_lat, vp_lon, vp_asn) in vps_coordinates.items():
            d = distance(vp_lat, center_lat, vp_lon, center_lon)

            if d > radius:
                continue

            # take at least one probe per AS
            if vp_asn not in selected_vps_per_asn:
                target_associated_vps.append(vp_addr)

                selected_vps_per_asn[vp_asn].append((vp_addr, vp_lat, vp_lon))

            else:
                # check if we already selected a VP in the same area (threshold)
                selected_close = False
                for _, selected_probe_lat, selected_probe_lon in selected_vps_per_asn[
                    vp_asn
                ]:
                    probe_distance = distance(
                        vp_lat, selected_probe_lat, vp_lon, selected_probe_lon
                    )

                    # do not select two VPs that are close together
                    if probe_distance < threshold:
                        selected_close = True
                        break

                if not selected_close:
                    target_associated_vps.append(vp_addr)

        return target_associated_vps

    def distance_vp_selection(
        self,
        target_addr: str,
        center_coordinates: tuple,
        vps_coordinate: list,
        ping_vps_to_target: dict,
        radius: int = 1000,
    ) -> [list, int]:
        """select the set of vps within a geographical area"""
        vp_selection = []
        measurement_cost = 0

        # TODO: make it so we always find at least one VP
        if not center_coordinates:
            return vp_selection, measurement_cost

        # get all vps within area of PoP
        target_associated_vps = self.get_vp_within_geographical_area(
            vps_coordinate, center_coordinates, radius
        )

        vp_selection = self.get_ping_to_target(
            target_associated_vps=target_associated_vps,
            ping_to_target=ping_vps_to_target,
        )

        return vp_selection, len(target_associated_vps)

    def multi_iteration(
        self,
        target_addr: str,
        vps_coordinate: dict,
        prev_vp_selection: list,
        ping_vps_to_target: dict,
    ) -> [list, int]:
        multi_iter_m_cost = 0
        closest_vp_addr, min_rtt = min(prev_vp_selection, key=lambda x: x[-1])

        # check if dns vp selection sufficient
        if min_rtt < self.latency_threshold:
            return prev_vp_selection, 0

        closest_vp_lat, closest_vp_lon, _ = vps_coordinate[closest_vp_addr]

        # If not, do selection for VP in geo area of VP with lowest rtt
        for _ in range(self.nb_round):
            # select probes close to the best vp
            radius = rtt_to_km(min_rtt)

            # increase radius to get at least one new VP, up to 1000 km added
            (
                vp_selection,
                _,
            ) = self.distance_vp_selection(
                target_addr=target_addr,
                center_coordinates=(closest_vp_lat, closest_vp_lon),
                vps_coordinate=vps_coordinate,
                ping_vps_to_target=ping_vps_to_target,
                radius=radius,
            )

            # in case we have too many VPs selected, simply return the previous results
            # we know that, in some cases, we cannot geolocate an IP address with precision
            if len(vp_selection) > 150:
                return prev_vp_selection, 0

            # remove duplicated measurement (we do not probe from a VP that probed already)
            vp_selection = list(set(prev_vp_selection + vp_selection))

            # get new closest VP and its coordinates
            new_closest_vp_addr, new_min_rtt = min(vp_selection, key=lambda x: x[-1])

            multi_iter_m_cost += len(vp_selection)

            # Stopping conditions
            # 1. same vp as before
            # 2. new rtt is greater than new one
            # 3. min rtt below threshold
            if (
                new_closest_vp_addr == closest_vp_addr
                or new_min_rtt >= min_rtt
                or new_min_rtt <= self.latency_threshold
            ):
                break

            # replace previous result with new one
            closest_vp_addr, min_rtt = (
                new_closest_vp_addr,
                new_min_rtt,
            )

        return vp_selection, multi_iter_m_cost

    def multi_iteration_cbg(
        self,
        vps_coordinate: dict,
        prev_vp_selection: list,
        ping_vps_to_target: dict,
    ) -> [list, int]:
        # perform CBG with results from previous stage
        centroid_lat, centroid_lon = cbg(
            vp_coordinates=vps_coordinate,
            vp_selection=prev_vp_selection,
        )

        # only get vps that are closed to the center of the centroid
        closest_vps = self.get_closest_vps(
            centroid_lat,
            centroid_lon,
            vps_coordinate,
            self.probing_budget,
        )

        # get ping results
        vp_selection = self.get_ping_to_target(
            target_associated_vps=[vp[0] for vp in closest_vps],
            ping_to_target=ping_vps_to_target,
        )

        # filter out vps that are in the same city/AS
        vp_selection = self.select_one_vp_per_as_city(vp_selection, vps_coordinate)

        return vp_selection, len(vp_selection)

    def multi_iteration_vp_selection(
        self,
        target_addr: str,
        vps_coordinate: dict,
        vps_per_subnet: dict,
        subnet_score: dict,
        ping_vps_to_target: dict,
        cbg_approximation: bool = False,
    ) -> [list, int]:
        """for a given target, select vps function of ecs-dns resolution"""
        vp_selection = []
        m_cost = 0

        target_subnet = get_prefix_from_ip(target_addr)

        # select vps using ECS-DNS algo
        (
            dns_vp_selection,
            dns_m_cost,
        ) = self.ecs_dns_vp_selection(
            vps_per_subnet=vps_per_subnet,
            ping_vps_to_target=ping_vps_to_target,
            target_subnet_score=subnet_score[target_subnet],
        )

        # TODO: change algo if ECS-DNS does not give results
        if not dns_vp_selection:
            return vp_selection, m_cost

        # After ECS-DNS VP selection, we perform a second measurement step.
        # Either we use the CBG area defined by the previous measurement, either we take the
        # single radius. Both of these methods are multi-iterative.
        if cbg_approximation:
            vp_selection, multi_iter_m_cost = self.multi_iteration_cbg(
                vps_coordinate=vps_coordinate,
                prev_vp_selection=dns_vp_selection,
                ping_vps_to_target=ping_vps_to_target,
            )

        else:
            vp_selection.extend(dns_vp_selection)
            vp_selection, multi_iter_m_cost = self.multi_iteration(
                target_addr=target_addr,
                vps_coordinate=vps_coordinate,
                prev_vp_selection=dns_vp_selection,
                ping_vps_to_target=ping_vps_to_target,
            )

        return vp_selection, dns_m_cost + multi_iter_m_cost

    async def select_vps_per_target(
        self, granularity: str = "answer_bgp_prefix"
    ) -> [dict, set]:
        """return a sorted list of pair (vp,rtt) per target using ECS-DNS algo"""
        # load datasets
        targets = load_json(self.path_settings.OLD_TARGETS)
        vps = load_json(self.path_settings.OLD_VPS)
        vps_per_subnet = self.get_vps_per_subnet(vps)
        vps_coordinate = self.get_vp_coordinates(vps)

        subnet_score = await self.get_subnet_score(targets, granularity)
        ping_vps_to_targets = await self.get_pings_per_target(
            self.clickhouse_settings.OLD_PING_VPS_TO_TARGET
        )

        logger.info(f"VP selection on {len(ping_vps_to_targets)} targets")

        target_vp_selection = defaultdict(list)
        target_m_cost = defaultdict(int)
        target_unmapped = set()

        for row in tqdm(ping_vps_to_targets):
            target_addr = row["target"]
            ping_vps_to_target = row["pings"]

            (
                target_vp_selection[target_addr],
                target_m_cost[target_addr],
            ) = self.multi_iteration_vp_selection(
                target_addr=target_addr,
                vps_coordinate=vps_coordinate,
                vps_per_subnet=vps_per_subnet,
                subnet_score=subnet_score,
                ping_vps_to_target=ping_vps_to_target,
            )

            if not target_vp_selection[target_addr]:
                target_unmapped.add(target_addr)

        return target_vp_selection, target_m_cost, target_unmapped

    async def select_vps_per_subnet(
        self, granularity: str = "answer_bgp_prefix"
    ) -> [dict, set]:
        """return a sorted list of pair (vp, median_rtt) per subnet ECS-DNS algo"""
        # load datasets
        vps = load_json(self.path_settings.OLD_VPS)
        vps_per_subnet = self.get_vps_per_subnet(vps)
        vps_coordinate = self.get_vp_coordinates(vps)

        ping_vps_to_subnet = await self.get_pings_per_subnet(
            self.clickhouse_settings.OLD_PING_VPS_TO_SUBNET
        )
        subnet_score = await self.get_subnet_score(targets, granularity)

        logger.info(f"VP selection on {len(ping_vps_to_subnet)} subnets")

        subnet_vp_selection = defaultdict(list)
        subnet_m_cost = defaultdict(int)
        subnet_unmapped = set()

        for subnet in ping_vps_to_subnet:
            min_rtts_per_vp = defaultdict(list)
            # get all vps rtt for the three representative, so we can calculate median error
            for target_addr, ping_vps_to_target in ping_vps_to_subnet[subnet].items():
                (
                    vp_selection,
                    m_cost,
                ) = self.multi_iteration_vp_selection(
                    target_addr=target_addr,
                    vps_coordinate=vps_coordinate,
                    vps_per_subnet=vps_per_subnet,
                    subnet_score=subnet_score,
                    ping_vps_to_target=ping_vps_to_target,
                )

                subnet_m_cost[subnet] += m_cost

                # get rtts to each representative for each subnet target
                for vp_addr, min_rtt in vp_selection:
                    min_rtts_per_vp[vp_addr].append(min_rtt)

            # Calculate median errors to the three representative and order VPs
            subnet_vp_selection[subnet] = self.get_best_subnet_vp(min_rtts_per_vp)

        return subnet_vp_selection, subnet_m_cost, subnet_unmapped


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(
        VPSelectionMultiIteration().main(
            targets=targets,
            vps=vps,
            output_path="multi_iter",
            target_selection=True,
            subnet_selection=True,
        )
    )
