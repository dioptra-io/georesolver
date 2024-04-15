from copy import deepcopy
from collections import defaultdict
from loguru import logger

from geogiant.common.files_utils import load_json, dump_json
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


def greedy_set_cover_algorithm(maximum_set_number: int, sets: dict):
    # Defensive copies

    sets_copy = deepcopy(sets)

    # Compute the union of all sets
    # Get all bgp prefixes
    space = set()
    for _, bgp_prefixes in sets_copy.items():
        space.update(bgp_prefixes)

    if maximum_set_number is None or maximum_set_number > len(sets_copy):
        maximum_set_number = len(sets_copy)

    selected_sets = []
    while len(selected_sets) < maximum_set_number and len(space) > 0:

        # Select the set with the biggest intersection
        best_set = set()
        best_set_elements = set()
        for hostname, bgp_prefixes in sets_copy.items():
            # The score of the a set if the score of the weights
            additional_elements = space.intersection(bgp_prefixes)

            if len(additional_elements) > len(best_set_elements):
                best_set = hostname
                best_set_elements = additional_elements

        selected_sets.append(best_set)
        space = space - best_set_elements

        del sets_copy[best_set]

    return selected_sets


def select_hostname_greedy_bgp(
    cdn_per_hostname: dict, bgp_prefix_per_hostname: dict
) -> dict:
    """select hostnames based on greedy selection algo"""
    hostname_greedy_bgp = greedy_set_cover_algorithm(
        maximum_set_number=1000,
        sets=bgp_prefix_per_hostname,
    )

    hostname_per_cdn_greedy_bgp = defaultdict(dict)
    for hostname in hostname_greedy_bgp:
        for org in cdn_per_hostname[hostname]:
            bgp_prefixes = bgp_prefix_per_hostname[hostname]
            hostname_per_cdn_greedy_bgp[org][hostname] = list(bgp_prefixes)

    return hostname_per_cdn_greedy_bgp


def select_hostname_greedy_per_cdn(hostname_per_cdn: dict) -> dict:
    """select hostnames based on greedy selection per CDN/org"""
    hostname_per_cdn_greedy_per_cdn = defaultdict(dict)
    for org, bgp_prefix_per_hostname in hostname_per_cdn.items():
        greedy_hostnames = greedy_set_cover_algorithm(10, bgp_prefix_per_hostname)
        greedy_hostnames = set(greedy_hostnames)

        for hostname in greedy_hostnames:
            bgp_prefixes = bgp_prefix_per_hostname[hostname]
            if not len(bgp_prefixes) > 10:
                continue

            hostname_per_cdn_greedy_per_cdn[org][hostname] = list(bgp_prefixes)

    return hostname_per_cdn_greedy_per_cdn


def select_hostname_max_bgp_prefix_per_cdn(
    hostname_per_cdn: dict, bgp_prefixes_per_hostname: dict
) -> dict:
    hostname_per_cdn_max_bgp_prefix = defaultdict(dict)
    for org, bgp_prefix_per_hostname in hostname_per_cdn.items():

        bgp_prefix_per_hostname = sorted(
            bgp_prefix_per_hostname.items(), key=lambda x: x[1]
        )
        for hostname, bgp_prefixes in bgp_prefix_per_hostname:

            # some hostnames has more than just one org
            hostname_bgp_prefixes = bgp_prefixes_per_hostname[hostname]

            if len(hostname_bgp_prefixes) > 20:
                hostname_per_cdn_max_bgp_prefix[org][hostname] = len(
                    hostname_bgp_prefixes
                )

    for org in hostname_per_cdn_max_bgp_prefix:
        hostname_per_cdn_max_bgp_prefix[org] = sorted(
            hostname_per_cdn_max_bgp_prefix[org].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.debug(f"ORGANIZATION:: {org}")
        for hostname, bgp_prefixes in hostname_per_cdn_max_bgp_prefix[org]:
            logger.debug(f"{hostname=}, {bgp_prefixes=}")

        hostname_per_cdn_max_bgp_prefix[org] = [
            hostname for hostname, _ in hostname_per_cdn_max_bgp_prefix[org]
        ]

    return hostname_per_cdn_max_bgp_prefix


def main() -> None:
    """
    based on ECS resolution on RIPE Atlas VPs, generate CSV files following different methodology:
    1) select greedy bgp prefixes (maximize coverage of bgp prefix)
    2) select greedy bgp prefixes per CDN (same but maximize bgp prefix coverage PER CDN)
    3) select hostnames that returned the maximum number of BGP prefixes, per CDN
    """
    cdn_per_hostname = load_json(
        path_settings.DATASET / "ecs_hostnames_organization.json"
    )

    bgp_prefix_per_hostname = defaultdict(set)
    for hostname, bgp_prefixes_per_cdn in cdn_per_hostname.items():
        for cdn, bgp_prefixes in bgp_prefixes_per_cdn.items():
            for bgp_prefix in bgp_prefixes:
                bgp_prefix_per_hostname[hostname].add(bgp_prefix)

    hostname_per_cdn = defaultdict(dict)
    for hostname in cdn_per_hostname:
        for org, bgp_prefixes in cdn_per_hostname[hostname].items():
            hostname_per_cdn[org][hostname] = bgp_prefixes

    # hostname_per_cdn_greedy_bgp = select_hostname_greedy_bgp(
    #     cdn_per_hostname,
    #     bgp_prefix_per_hostname,
    # )
    # hostname_per_cdn_greedy_per_cdn = select_hostname_greedy_per_cdn(hostname_per_cdn)

    hostname_per_cdn_max_bgp_prefix = select_hostname_max_bgp_prefix_per_cdn(
        hostname_per_cdn, bgp_prefix_per_hostname
    )

    # dump_json(
    #     data=hostname_per_cdn_greedy_bgp,
    #     output_file=path_settings.DATASET / "hostname_per_cdn_greedy_bgp.json",
    # )

    # dump_json(
    #     data=hostname_per_cdn_greedy_per_cdn,
    #     output_file=path_settings.DATASET / "hostname_per_cdn_greedy_cdn.json",
    # )

    # > X bgp prefixes (geo diversity) -> do not discriminate
    # merge main orga (AMAZON-AES / AMAZON-02)
    # 1. take main orga per hostname in case of multi -> hostname to main orga (orga with max bgp prefixes)
    # arg: how does redirection?
    # + merge orga ()

    dump_json(
        data=hostname_per_cdn_max_bgp_prefix,
        output_file=path_settings.DATASET / "hostname_per_cdn_max_bgp_prefix.json",
    )


if __name__ == "__main__":
    main()
