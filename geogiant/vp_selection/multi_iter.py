"""main functions for analyzing geolocation results"""
import math
from collections import defaultdict
from multiprocessing import Pool

from loguru import logger
from tqdm import tqdm

from common.files_utils import load_json
from common.ip_addresses_utils import get_prefix_from_ip
from common.settings import PathSettings
from common.geoloc import cbg, rtt_to_km
from vp_selection_algo import VPSelectionDistance, VPSelectionDNS

path_settings = PathSettings()


class VPSelectionMultiIteration(VPSelectionDNS, VPSelectionDistance):
    """mixed vp selection, first use ECS-DNS resolution for selection
    then use geographical location around vp with the lowest latency
    """

    latency_threshold: int = 1
    nb_round: int = 1
    probing_budget: int = 30

    def greedy_selection_probes_impl(self, probe, distance_per_probe, selected_probes):
        distances_log = [
            math.log(distance_per_probe[p])
            for p in selected_probes
            if p in distance_per_probe and distance_per_probe[p] > 0
        ]
        total_distance = sum(distances_log)
        return probe, total_distance

    def maximize_coverage(self, vp_selection: list, vp_distance_matrix: dict):
        selected_probes = []
        remaining_probes = vp_selection
        with Pool(12) as p:
            while (
                len(remaining_probes) > 0 and len(selected_probes) < self.probing_budget
            ):
                args = []
                for vp in remaining_probes:
                    args.append((vp, vp_distance_matrix[vp[0]], selected_probes))

                results = p.starmap(self.greedy_selection_probes_impl, args)

                furthest_probe_from_selected, _ = max(results, key=lambda x: x[1])
                selected_probes.append(furthest_probe_from_selected)
                remaining_probes.remove(furthest_probe_from_selected)

        return selected_probes

    def multi_iteration(
        self,
        target_addr: str,
        vps_coordinate: dict,
        prev_vp_selection: list,
        ping_vps_to_target: dict,
        vp_distance_matrix: dict,
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

            # TODO: avoid doing that shit
            for _ in range(3):
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

                if len(vp_selection) > 1000:
                    return prev_vp_selection, 0

                if vp_selection:
                    break

                radius += 50

            # if no new vps_coordinate are found, stop
            # if not vp_selection:
            #     break

            # remove duplicated measurement (we do not probe from a VP that probed already)
            vp_selection = list(set(prev_vp_selection + vp_selection))

            # get new closest VP and its coordinates
            new_closest_vp_addr, new_min_rtt = min(vp_selection, key=lambda x: x[-1])

            # filter so we are not probing from too many vps, in case of inflated RTT
            # if len(vp_selection) > self.probing_budget:
            #     # vp_selection = self.maximize_coverage(vp_selection, vp_distance_matrix)

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

    ################################################################
    # NEW CODE                                                     #
    ################################################################
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
        vp_distance_matrix: dict,
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

        vp_selection, multi_iter_m_cost = self.multi_iteration(
            target_addr=target_addr,
            vps_coordinate=vps_coordinate,
            prev_vp_selection=dns_vp_selection,
            ping_vps_to_target=ping_vps_to_target,
            vp_distance_matrix=vp_distance_matrix,
        )

        # vp_selection, multi_iter_m_cost = self.multi_iteration_cbg(
        #     vps_coordinate=vps_coordinate,
        #     prev_vp_selection=dns_vp_selection,
        #     ping_vps_to_target=ping_vps_to_target,
        # )

        # vp_selection.extend(dns_vp_selection)

        return vp_selection, dns_m_cost + multi_iter_m_cost

    def select_vps_per_target(self, round: int = 0) -> [dict, set]:
        """return a sorted list of pair (vp,rtt) per target using ECS-DNS algo"""
        # load datasets
        targets = load_json(self.path_settings.TARGETS)
        vps = load_json(self.path_settings.VPS)
        vps_per_subnet = self.get_vps_per_subnet(vps)
        subnet_score = self.get_subnet_score(targets)
        ping_vps_to_targets = self.get_meshed_pings_targets()
        vps_coordinate = self.get_vp_coordinates(vps)
        ping_vps_to_targets = self.get_meshed_pings_targets()
        vp_distance_matrix = load_json(self.path_settings.VPS_DISTANCE_MATRIX)

        logger.info(f"VP selection on {len(ping_vps_to_targets)} targets")

        target_vp_selection = defaultdict(list)
        target_m_cost = defaultdict(int)
        target_unmapped = set()

        for target_addr, ping_vps_to_target in tqdm(ping_vps_to_targets.items()):
            (
                target_vp_selection[target_addr],
                target_m_cost[target_addr],
            ) = self.multi_iteration_vp_selection(
                target_addr=target_addr,
                vps_coordinate=vps_coordinate,
                vps_per_subnet=vps_per_subnet,
                subnet_score=subnet_score,
                ping_vps_to_target=ping_vps_to_target,
                vp_distance_matrix=vp_distance_matrix,
            )

            if not target_vp_selection[target_addr]:
                target_unmapped.add(target_addr)

        return target_vp_selection, target_m_cost, target_unmapped

    def select_vps_per_subnet(self) -> [dict, set]:
        """return a sorted list of pair (vp, median_rtt) per subnet ECS-DNS algo"""
        # load datasets
        vps = load_json(self.path_settings.VPS)
        ping_vps_to_subnet = self.get_meshed_pings_subnets()
        subnet_assigned_pops = self.get_assigned_pops(
            table_name=self.clickhouse_settings.DNS_MAPPING_PERIODIC_TABLE,
            hostname_filter=["google"],
        )

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
                    subnet_assigned_pops=subnet_assigned_pops,
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
    targets = load_json(path_settings.TARGETS)
    vps = load_json(path_settings.VPS)

    VPSelectionMultiIteration().main(
        targets=targets,
        vps=vps,
        output_path="multi_iteration",
        target_selection=True,
        subnet_selection=False,
    )
