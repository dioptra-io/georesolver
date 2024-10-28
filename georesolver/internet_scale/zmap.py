import subprocess

from random import shuffle
from tqdm import tqdm
from loguru import logger
from tqdm import tqdm
from pathlib import Path

from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.clickhouse.queries import (
    load_vps,
    get_pings_per_target,
)
from georesolver.common.files_utils import (
    load_csv,
    load_json,
    dump_json,
    create_tmp_csv_file,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

PROBING_BUDGET = 50
PING_GEORESOLVER_TABLE = "pings_internet_scale"
PING_AGGREGATION_TABLE = "pings_internet_scale_aggregation"
INTERNET_SCALE_SUBNETS_PATH = (
    path_settings.INTERNET_SCALE_DATASET / "all_internet_scale_subnets.json"
)
VPS_SUBNET_PATH = path_settings.DATASET / "vps_subnets_filtered.json"
INTERNET_SCALE_EVALUATION_PATH = (
    path_settings.RESULTS_PATH
    / "internet_scale_evaluation/results__routers_internet_scale.pickle"
)
INTERNET_SCALE_RESPONSIVE_IP_ADDRS = (
    path_settings.INTERNET_SCALE_DATASET / "responsive_ip_addrs_per_subnet.json"
)


def get_shortest_ping_vps(
    ping_table: str, removed_vps: list[str] = [], nb_vps: int = 10
) -> None:
    """for each IP addresses retrieved the shortest ping"""
    ping_vps_to_target = get_pings_per_target(ping_table, removed_vps)

    shortest_ping_per_target = []
    for target_addr, target_pings in tqdm(ping_vps_to_target.items()):

        target_pings = sorted(target_pings, key=lambda x: x[-1])
        shortest_ping_rtt = target_pings[0][1]

        if shortest_ping_rtt > 2:
            continue

        vps = [vp for vp, _ in target_pings[:nb_vps]]

        shortest_ping_per_target.append((target_addr, vps))

    return shortest_ping_per_target


def get_responsive_ip_addrs_in_subnet(subnet: str) -> Path:
    subnet += "/24"

    tmp_file_path = create_tmp_csv_file([])
    zmap_cmd = f"echo {path_settings.SUDO_PWD} | sudo -S zmap --probe-module=icmp_echoscan -p 0 -r 10000 -N 50 {subnet} -O csv --output-file={tmp_file_path}"

    logger.debug(f"ZMap cmd:: {zmap_cmd}")

    results = subprocess.run(
        args=zmap_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )

    if results.stderr:
        pass

    responsive_ip_address = load_csv(tmp_file_path)

    logger.debug(f"Responsive IP addresses in {subnet}:: {len(responsive_ip_address)}")

    tmp_file_path.unlink()

    return responsive_ip_address


def zmap() -> None:
    # load targets, vps and cached measurements
    vps = load_vps(clickhouse_settings.VPS_FILTERED_TABLE)
    targets = get_shortest_ping_vps(PING_GEORESOLVER_TABLE)
    cached_pings = get_shortest_ping_vps(PING_AGGREGATION_TABLE)
    cached_subnets = [get_prefix_from_ip(target) for target, _ in cached_pings]

    # filter targets with cached pings
    if cached_subnets:
        filtered_targets = []
        for target, ecs_vps in targets:
            target_subnet = get_prefix_from_ip(target)
            if target_subnet in cached_subnets:
                continue

            filtered_targets.append((target, ecs_vps))

        targets = filtered_targets

    logger.info(f"Targets in schedule:: {len(targets)} subnets")

    shuffle(targets)

    vp_id_per_addr = {}
    for vp in vps:
        vp_id_per_addr[vp["addr"]] = vp["id"]

    batch_size_zmap = 1_00
    for i in range(0, len(targets), batch_size_zmap):
        logger.info(
            f"ZMAP, batch:: {(i +batch_size_zmap) // batch_size_zmap}/{len(targets) //batch_size_zmap}"
        )
        measurement_schedule = []
        for target, ecs_vps in targets[i : i + batch_size_zmap]:
            subnet = get_prefix_from_ip(target)
            responsive_ip_addrs = get_responsive_ip_addrs_in_subnet(subnet)

            for ip_addr in responsive_ip_addrs[:50]:
                vp_ids = []
                for vp_addr in ecs_vps:
                    try:
                        vp_ids.append(vp_id_per_addr[vp_addr])
                    except KeyError:
                        continue

                measurement_schedule.append((ip_addr, vp_ids))

        if INTERNET_SCALE_RESPONSIVE_IP_ADDRS.exists():
            cached_schedule = load_json(INTERNET_SCALE_RESPONSIVE_IP_ADDRS)
            if cached_schedule:
                measurement_schedule.extend(cached_schedule)

        dump_json(measurement_schedule, INTERNET_SCALE_RESPONSIVE_IP_ADDRS)


if __name__ == "__main__":
    zmap()
