"""This script is meant to evaluate if IP addresses within the same prefix are geographically close together"""

from loguru import logger
from pprint import pformat
from random import shuffle
from collections import defaultdict

from georesolver.scheduler import create_agents
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.files_utils import (
    load_json,
    dump_json,
    load_iter_csv,
    dump_csv,
)
from georesolver.common.settings import PathSettings

path_settings = PathSettings()

CONFIG_PATH = path_settings.DATASET / "experiment_config/subnet_aggregation_config.json"
TARGET_FILE = (
    path_settings.DATASET / "subnet_aggregation/subnet_aggregation_targets.csv"
)


def load_addr_per_subnet() -> None:
    """get all ZMap addr per prefix"""
    # load zmap file
    zmap_addrs = load_iter_csv(path_settings.DATASET / "zmap_icmp_scan_2024_09.csv")

    addr_per_subnet = defaultdict(list)
    for ip_addr in zmap_addrs:
        ip_prefix = get_prefix_from_ip(ip_addr)
        addr_per_subnet[ip_prefix].append(ip_addr)

    # dump file
    dump_json(
        addr_per_subnet,
        path_settings.DATASET / "subnet_aggregation/addr_per_subnet.json",
    )


def get_random_sample_per_prefix(dataset_size: int) -> None:
    """take a random sample of prefixes"""
    addr_per_prefix = 25
    logger.info("Loading random samples per subnets")

    # get random dataset of IP addresses
    addr_per_subnet = load_json(
        path_settings.DATASET / "subnet_aggregation/addr_per_subnet.json"
    )
    targets = set()
    prefix_count = 0
    for _, addrs in addr_per_subnet.items():
        if not len(addrs) > addr_per_prefix:
            continue

        # get random sample per prefix
        shuffle(addrs)
        targets.update(addrs[:addr_per_prefix])

        prefix_count += 1
        if prefix_count >= dataset_size:
            break

    dump_csv(targets, TARGET_FILE)


def load_target_file(dataset_size: int = 10_000) -> None:
    """generate target file for subnet aggregation experiment"""
    if not TARGET_FILE.exists():
        if not (
            path_settings.DATASET / "subnet_aggregation/addr_per_subnet.json"
        ).exists():
            load_addr_per_subnet()

        get_random_sample_per_prefix(dataset_size)

    else:
        logger.info(f"Target file:: {TARGET_FILE} already exits")


def run_experiment() -> None:
    """load config and start experiment"""
    config = load_json(CONFIG_PATH)
    logger.info("Running subnet aggregation experiment from config:: ")
    logger.info("config=\n{}", pformat(config))

    agents = create_agents(CONFIG_PATH)
    for agent in agents:
        agent.run()


def main() -> None:
    """
    measurement entry point:
        - generate target file:
            - takes N /24 subnets with at least 25% responsive IP addresses
            - takes /24 subnets that belongs to larger prefixes (in MaxMind dataset)
        - run georesolver measurement
    """
    load_target_file()
    run_experiment()


if __name__ == "__main__":
    main()
