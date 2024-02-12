"""main functions for analyzing geolocation results"""

import asyncio

from pathlib import Path
from numpy import mean, median
from collections import defaultdict
from loguru import logger
from abc import ABC, abstractmethod
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import GetPingsPerTarget, GetAvgRTTPerSubnet, GetPingsPerSubnet

from geogiant.common.geoloc import distance
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.files_utils import load_json, dump_pickle
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()


class VPSelectionBase(ABC):
    clickhouse_settings = ClickhouseSettings()
    path_settings = PathSettings()

    @classmethod
    async def get_pings_per_target(self, table_name: str) -> dict:
        """
        return meshed ping for all targets
        ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
        """
        async with AsyncClickHouseClient(
            **self.clickhouse_settings.clickhouse
        ) as client:
            resp = await GetPingsPerTarget().execute(
                client=client,
                table_name=table_name,
            )

        return resp

    @classmethod
    async def get_pings_per_target_parsed(self, table_name: str) -> dict:
        """
        return meshed ping for all targets
        ping_vps_to_target[target_addr] = [(vp_addr, min_rtt)]
        """
        ping_vps_to_target = {}
        async with AsyncClickHouseClient(
            **self.clickhouse_settings.clickhouse
        ) as client:
            resp = await GetPingsPerTarget().execute(
                client=client,
                table_name=table_name,
            )

        for row in resp:
            ping_vps_to_target[row["target"]] = row["pings"]

        return resp

    @classmethod
    async def get_pings_per_subnet(self, table_name: str) -> dict[dict]:
        """
        get meshed pings between targets and vps
        ping_vps_to_subnet[subnet][target_addr] = [(vp_addr, min_rtt)]
        """
        ping_vps_to_subnet = defaultdict(dict)
        async with AsyncClickHouseClient(
            **self.clickhouse_settings.clickhouse
        ) as client:
            resp = await GetPingsPerSubnet().execute(
                client=client,
                table_name=table_name,
            )

        for row in resp:
            target = row["target"]
            subnet = row["subnet"]
            ping_vps_to_subnet[subnet][target] = row["ping_to_target"]

        return ping_vps_to_subnet

    @classmethod
    async def get_avg_rtt_per_subnet(self, table_name: str) -> dict[dict]:
        """
        get meshed pings between targets and vps
        ping_vps_to_subnet[subnet][target_addr] = [(vp_addr, min_rtt)]
        """
        async with AsyncClickHouseClient(
            **self.clickhouse_settings.clickhouse
        ) as client:
            resp = await GetAvgRTTPerSubnet().execute(
                client=client,
                table_name=table_name,
            )

        return resp

    @classmethod
    def get_ping_to_target(
        self,
        target_associated_vps: list,
        ping_to_target: list,
    ) -> list:
        vp_selection = []

        # filter out all vps not included by ecs-dns methodology
        for vp_addr, min_rtt in ping_to_target:
            if vp_addr in target_associated_vps:
                vp_selection.append((vp_addr, min_rtt))

        return vp_selection

    @classmethod
    def get_coordinates_from_addr(
        self,
        target_addr: str,
        targets: list,
    ) -> tuple:
        """return coordinates based on target IP address"""
        # get target geo
        for target in targets:
            if target_addr == target["address_v4"]:
                return target["geometry"]["coordinates"]

        return None, None

    @classmethod
    def get_vp_from_addr(
        self,
        vp_addr: str,
        vps: list,
    ) -> dict:
        for vp in vps:
            if vp_addr == vp["address_v4"]:
                return vp["geometry"]["coordinates"]

        return None, None

    @classmethod
    def get_best_subnet_vp(self, min_rtt_per_vp: dict) -> list:
        """return a list of selected VPs ordered by median error"""
        # calculate all median errors
        median_rtt_subnet = []
        for vp_addr, min_rtts in min_rtt_per_vp.items():
            median_rtt_subnet.append((vp_addr, median(min_rtts)))

        # sort median error for validation
        subnet_best_vps = sorted(median_rtt_subnet, key=lambda x: x[-1])

        return subnet_best_vps

    @classmethod
    def get_vp_coordinates(self, vps: list) -> dict:
        """create a dictionary of vps coordinates for optimized retrieval"""
        # get all vp coordinates
        vps_coordinates = dict()
        for vp in vps:
            vp_addr = vp["address_v4"]
            vp_lon, vp_lat = vp["geometry"]["coordinates"]
            vp_asn = vp["asn_v4"]

            vps_coordinates[vp_addr] = (vp_lat, vp_lon, vp_asn)

        return vps_coordinates

    @classmethod
    def get_vps_per_subnet(self, vps: list) -> dict:
        """create a dictionary of vps coordinates for optimized retrieval"""
        vps_subnet = defaultdict(list)
        for vp in vps:
            vp_addr = vp["address_v4"]
            vp_lon, vp_lat = vp["geometry"]["coordinates"]
            vp_asn = vp["asn_v4"]
            vp_subnet = get_prefix_from_ip(vp_addr)
            vps_subnet[vp_subnet].append((vp_addr, vp_lat, vp_lon, vp_asn))
        return vps_subnet

    @classmethod
    def get_parsed_associated_vps(
        self,
        highest_score_subnets: list,
        vps_per_subnet: dict,
    ) -> tuple[list, dict]:
        """parse list of vps per subnet to get a list of vp addrs and their coordinates"""
        # get vps addr
        vp_addrs = []
        vp_coordinates = {}
        for subnet, _ in highest_score_subnets:
            vp_addrs.extend([vp_addr for vp_addr, _, _, _ in vps_per_subnet[subnet]])

            # get vp coordinates per addr for filtering vps in same city/AS
            for vp_addr, vp_lat, vp_lon, vp_asn in vps_per_subnet[subnet]:
                vp_coordinates[vp_addr] = (vp_lat, vp_lon, vp_asn)

        return vp_addrs, vp_coordinates

    @classmethod
    def get_closest_vps(
        self,
        lat: float,
        lon: float,
        vps_coordinate: dict,
        probing_budget: int,
    ) -> list:
        """from a tuple of coordinates return the vps that are the closest to it"""

        closest_vps = []
        for vp_addr, (vp_lat, vp_lon, _) in vps_coordinate.items():
            d_to_centroid = distance(
                vp_lat,
                lat,
                vp_lon,
                lon,
            )

            closest_vps.append((vp_addr, vp_lat, vp_lon, d_to_centroid))

        # sort vps function of the distance to the coordinates
        closest_vps = sorted(closest_vps, key=lambda x: x[-1])
        closest_vps = closest_vps[:probing_budget]

        return closest_vps

    @classmethod
    def select_one_vp_per_as_city(
        self,
        raw_vp_selection: list,
        vp_coordinates: dict,
        threshold: int = 20,
    ) -> None:
        """from a list of VP, select one per AS and per city"""
        filtered_vp_selection = []
        selected_vps_per_asn = defaultdict(list)

        for vp_addr, min_rtt in raw_vp_selection:
            vp_lat, vp_lon, vp_asn = vp_coordinates[vp_addr]

            # take at least one probe per AS
            if vp_asn not in selected_vps_per_asn:
                filtered_vp_selection.append((vp_addr, min_rtt))

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
                    filtered_vp_selection.append((vp_addr, min_rtt))

        return filtered_vp_selection

    @classmethod
    def geoloc_error_target(self, targets: dict, vps: dict, vp_selection: dict) -> dict:
        """from a set of targets, vps and vp selection algo results,
        return the estimated error for each target
        """
        geolocation_errors = dict()
        min_latencies = dict()

        logger.info(f"Geolocation error for {len(targets)} targets")

        for target in targets:
            target_lon, target_lat = target["geometry"]["coordinates"]

            # get target subnet allocated vp
            try:
                target_vp_selection = vp_selection[target["address_v4"]]
                target_vp_selection = sorted(target_vp_selection, key=lambda x: x[-1])
            except KeyError:
                logger.error(f"no vp selection for target: {target['address_v4']}")
                continue

            # TODO: use min rtt for precision estimation
            vp_lat = vp_lon = None

            for vp_addr, min_rtt in target_vp_selection:
                vp_lon, vp_lat = self.get_vp_from_addr(vp_addr, vps)
                if vp_lon and vp_lat:
                    break

            if not vp_lon or not vp_lat:
                continue

            # compute error distance between target and selected vp
            geolocation_errors[target["address_v4"]] = distance(
                target_lat, vp_lat, target_lon, vp_lon
            )
            min_latencies[target["address_v4"]] = min_rtt

        return geolocation_errors, min_latencies

    @classmethod
    def geoloc_error_subnet(self, targets: dict, vps: dict, vp_selection: dict) -> dict:
        """from a set of targets, vps and vp selection algo results,
        return the estimated error for each target
        """
        geolocation_errors = dict()
        min_latencies = dict()

        logger.info(f"Geolocation error for {len(targets)} targets")

        # easy access to vps coordinates
        parsed_vps = {}
        for vp in vps:
            parsed_vps[vp["address_v4"]] = vp["geometry"]["coordinates"]

        for target in targets:
            target_lon, target_lat = target["geometry"]["coordinates"]

            # get target subnet allocated vp
            target_subnet = get_prefix_from_ip(target["address_v4"])

            try:
                subnet_vp_selection = vp_selection[target_subnet]
            except KeyError:
                # logger.error("subnet missing")
                continue

            # TODO: use min rtt for precision estimation
            vp_lat = vp_lon = None
            for vp_addr, min_rtt in subnet_vp_selection:
                vp_lon, vp_lat = self.get_vp_from_addr(vp_addr, vps)
                if vp_lon and vp_lat:
                    break

            if not vp_lon or not vp_lat:
                logger.debug(f"No VPs found for target: {target['address_v4']}")
                continue

            # compute error distance between target and selected vp
            geolocation_errors[target["address_v4"]] = distance(
                target_lat, vp_lat, target_lon, vp_lon
            )

            min_latencies[target["address_v4"]] = min_rtt

        return geolocation_errors, min_latencies

    @abstractmethod
    async def select_vps_per_target(self) -> [dict, set]:
        """for each targets, select a set of vps to perform measurements"""
        raise NotImplementedError(
            f"Function: {__name__} must be implemented for class: {__class__}"
        )

    @abstractmethod
    async def select_vps_per_subnet(self) -> [dict, set]:
        """for each targets, select a set of vps to perform measurements"""
        raise NotImplementedError(
            f"Function: {__name__} must be implemented for class: {__class__}"
        )

    async def main(
        self,
        targets: list,
        vps: list,
        output_path: Path,
        target_selection: bool = False,
        subnet_selection: bool = False,
        **kwargs,
    ) -> None:
        """perform vp selection on for a set of given targets"""
        # perform vp selection algo, retrieve measurement cost as well
        if target_selection:
            (
                vp_target_selection,
                measurement_cost,
                target_unmapped,
            ) = await self.select_vps_per_target(**kwargs)

            logger.info(
                f"vp selection for : {sum([1 for vp_selection in vp_target_selection.values() if vp_selection])} targets using direct method"
            )

            if measurement_cost and target_unmapped:
                logger.info(f"target not geolocated: {len(target_unmapped)}")
                logger.info(
                    f"Mean measurement cost per target: {mean([cost for cost in measurement_cost.values()])}"
                )

            # evaluate geolocation error
            geoloc_error_target, min_latencies = self.geoloc_error_target(
                targets=targets,
                vps=vps,
                vp_selection=vp_target_selection,
            )

            logger.info(
                f"found geolocation for : {len(geoloc_error_target)} targets using direct method"
            )

            dump_pickle(
                vp_target_selection,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_target_vp_selection.pickle",
            )

            dump_pickle(
                measurement_cost,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_target_measurement_cost.pickle",
            )

            dump_pickle(
                geoloc_error_target,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_target_geoloc_error.pickle",
            )

            dump_pickle(
                min_latencies,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_target_min_latencies.pickle",
            )

            return geoloc_error_target, min_latencies

        if subnet_selection:
            (
                vp_subnet_selection,
                measurement_cost,
                subnet_unmapped,
            ) = await self.select_vps_per_subnet()

            logger.info(
                f"vp selection for : {sum([1 for vp_selection in vp_subnet_selection.values() if len(vp_selection) > 0])} targets using closest vp"
            )

            # evaluate geolocation error
            geoloc_error_subnet, min_latencies = self.geoloc_error_subnet(
                targets=targets,
                vps=vps,
                vp_selection=vp_subnet_selection,
            )

            logger.info(
                f"found geolocation for : {len(geoloc_error_subnet)} targets using closest vp"
            )

            dump_pickle(
                subnet_selection,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_subnet_vp_selection.pickle",
            )

            dump_pickle(
                measurement_cost,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_subnet_measurement_cost.pickle",
            )

            dump_pickle(
                geoloc_error_subnet,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_subnet_geoloc_error.pickle",
            )

            dump_pickle(
                min_latencies,
                self.path_settings.RESULTS_PATH
                / f"{output_path}_subnet_min_latencies.pickle",
            )


class VPSelectionReference(VPSelectionBase):
    async def select_vps_per_target(self) -> list:
        """return a sorted list of pair (vp,rtt) per target"""
        target_vp_selection = dict()
        ping_vps_to_target = await self.get_pings_per_target(
            self.clickhouse_settings.OLD_PING_VPS_TO_TARGET
        )

        for row in ping_vps_to_target:
            target = row["target"]
            pings = row["pings"]
            target_vp_selection[target] = sorted(pings, key=lambda x: x[-1])

        return target_vp_selection, None, None

    async def select_vps_per_subnet(
        self,
    ) -> dict[str, list]:
        """return a sorted list of pair (vp, median_rtt) per subnet"""
        ping_vps_to_subnet = await self.get_avg_rtt_per_subnet(
            self.clickhouse_settings.OLD_PING_VPS_TO_SUBNET
        )

        logger.info(f"VP selection on {len(ping_vps_to_subnet)} subnets")

        subnet_vp_selection = defaultdict(list)
        for row in ping_vps_to_subnet:
            subnet = row["subnet"]
            vps_avg_rtt = row["vps_avg_rtt"]

            subnet_vp_selection[subnet] = vps_avg_rtt

        return subnet_vp_selection, None, None


if __name__ == "__main__":
    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    asyncio.run(
        VPSelectionReference().main(
            targets=targets,
            vps=vps,
            output_path="ref",
            target_selection=True,
            subnet_selection=False,
        )
    )
