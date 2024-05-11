from tqdm import tqdm
from collections import defaultdict
from loguru import logger

from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.files_utils import dump_json
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


def ip_addresses_hitlist() -> None:
    targets_per_prefix = defaultdict(list)

    total_addr = set()
    total_subnets = set()
    with open(path_settings.ADDRESS_FILE, "r") as f:
        for row in tqdm(f.readlines()[1:]):
            row = row.split("\t")

            try:
                score = row[1]
            except IndexError:
                continue

            if int(score) < 50:
                continue

            ip_addr = row[-1].strip("\n")
            subnet = get_prefix_from_ip(ip_addr)

            targets_per_prefix[subnet].append(ip_addr)

            total_addr.add(ip_addr)
            total_subnets.add(subnet)

    logger.info(f"{len(total_addr)=}")
    logger.info(f"{len(total_subnets)=}")

    dump_json(targets_per_prefix, path_settings.USER_HITLIST_FILE)


if __name__ == "__main__":
    generate_hitlist = True

    if generate_hitlist:
        ip_addresses_hitlist()
