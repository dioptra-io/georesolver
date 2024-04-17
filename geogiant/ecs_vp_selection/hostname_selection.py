from copy import deepcopy
from collections import defaultdict
from loguru import logger
from pych_client import ClickHouseClient

from geogiant.clickhouse import GetAllNameServers
from geogiant.common.files_utils import load_csv, load_json, dump_json
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_all_name_servers() -> dict:
    """retrieve all unparsed name servers"""
    name_servers_per_hostname = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetAllNameServers().execute(client, "name_servers")

        for row in rows:
            name_servers_per_hostname[row["hostname"]] = row["name_servers"]

    return name_servers_per_hostname


def parse_name_servers(name_servers_per_hostname: dict, tlds: list[str]) -> dict:
    """parse name server to extract main organization behin them"""
    parsed_name_servers_per_hostname = {}
    for hostname, name_servers in name_servers_per_hostname.items():
        parsed_name_servers = set()
        for name_server in name_servers:

            name_server = name_server.split(".")

            if name_server[-1] == "":
                parsed_name_server = name_server[-3]
            elif len(name_server) == 1:
                parsed_name_server = name_server[0]
            else:
                parsed_name_server = name_server[-2]

            if "awsdns" in name_server:
                parsed_name_server = "awsdns"

            if "msedge" in name_server:
                parsed_name_server = "msedge"

            if "bunny" in name_server:
                parsed_name_server = "bunny"

            if len(parsed_name_server) <= 3:

                # filter all tlds from name servers
                for tld in tlds:
                    try:
                        name_server.remove(tld)
                    except ValueError:
                        continue

                if not name_server:
                    continue

                if name_server[-1] == "":
                    parsed_name_server = name_server[-2]
                else:
                    parsed_name_server = name_server[-1]

                if "awsdns" in parsed_name_server:
                    parsed_name_server = "awsdns"

                if "msedge" in parsed_name_server:
                    parsed_name_server = "msedge"

                if "bunny" in parsed_name_server:
                    parsed_name_server = "bunny"

                if "gslb" in parsed_name_server:
                    parsed_name_server = "gslb"

                if len(parsed_name_server) <= 3:
                    continue

                # manually remove ns identified as wrong
                if parsed_name_server in ["customer", "invalid", "gslb", "president"]:
                    continue

                if "ns" in parsed_name_server:
                    continue

                # just for clarity
                if parsed_name_server in ["awsdns", "cloudtimes", "revolutionise"]:
                    continue

            parsed_name_servers.add(parsed_name_server)

        parsed_name_servers_per_hostname[hostname] = parsed_name_servers

    return parsed_name_servers_per_hostname


def get_hostname_per_name_server(parsed_name_servers: dict) -> list:
    """get all hostnames that have a unique name server"""
    hostname_per_name_server = defaultdict(list)
    for hostname, name_server in parsed_name_servers.items():

        if len(name_server) != 1:
            continue
        else:
            name_server = list(name_server)[0]

        hostname_per_name_server[name_server].append(hostname)

    hostname_per_name_server = sorted(
        hostname_per_name_server.items(), key=lambda x: len(x[-1]), reverse=True
    )

    return hostname_per_name_server


def get_hostname_per_main_org(
    bgp_per_cdn_per_hostname: dict, main_org_threshold: float = 0.6
) -> list:
    """return the main hosting organization for each hostname"""
    no_main_org = set()
    hostname_per_main_org = defaultdict(set)
    bgp_prefixes_per_hostname = defaultdict(set)
    for hostname, bgp_per_cdn in bgp_per_cdn_per_hostname.items():
        total_bgp_prefixes = set()
        for bgp_prefixes in bgp_per_cdn.values():
            total_bgp_prefixes.update(bgp_prefixes)
            bgp_prefixes_per_hostname[hostname].update(bgp_prefixes)

        main_cdn, max_cdn_bgp_prefixes = max(
            bgp_per_cdn.items(), key=lambda x: len(x[1])
        )

        frac_bgp_prefixes_main_cdn = len(max_cdn_bgp_prefixes) / len(total_bgp_prefixes)

        if frac_bgp_prefixes_main_cdn > main_org_threshold:
            hostname_per_main_org[main_cdn].add(hostname)
        else:
            no_main_org.add(hostname)

    hostname_per_main_org = sorted(
        hostname_per_main_org.items(), key=lambda x: len(x[1]), reverse=True
    )

    return hostname_per_main_org


def parse_hostname_per_main_org(
    hostname_per_main_org: list,
    bgp_prefixes_per_hostname: dict,
    bgp_prefix_threshold: int = 2,
) -> dict:
    """merge hosting organizations if they belong to the same company under different names"""

    hostname_per_merged_main_orgs = defaultdict(list)
    for main_org, hostnames in hostname_per_main_org:
        for hostname in hostnames:
            bgp_prefixes = bgp_prefixes_per_hostname[hostname]

            if len(bgp_prefixes) > bgp_prefix_threshold:

                if "AMAZON" in main_org:
                    hostname_per_merged_main_orgs["AMAZON"].append(hostname)

                elif "GOOGLE" in main_org:
                    hostname_per_merged_main_orgs["GOOGLE"].append(hostname)

                elif "AKAMAI" in main_org:
                    hostname_per_merged_main_orgs["AKAMAI"].append(hostname)

                elif "APPLE" in main_org:
                    hostname_per_merged_main_orgs["APPLE"].append(hostname)

                elif "MICROSOFT" in main_org:
                    hostname_per_merged_main_orgs["MICROSOFT"].append(hostname)

                elif "TENCENT" in main_org:
                    hostname_per_merged_main_orgs["TENCENT"].append(hostname)

                elif "CHINANET" in main_org:
                    hostname_per_merged_main_orgs["CHINANET"].append(hostname)

                elif "CMNET" in main_org:
                    hostname_per_merged_main_orgs["CMNET"].append(hostname)

                else:
                    hostname_per_merged_main_orgs[main_org].append(hostname)

    return hostname_per_merged_main_orgs


def get_hostname_per_org_per_ns(
    hostname_per_name_server: list,
    main_org_per_hostname: dict,
    bgp_prefixes_per_hostname: dict,
) -> dict[dict]:
    """sort hostnames per org per name servers"""
    hostname_per_org_per_name_servers = defaultdict(dict)
    for name_server, hostnames in hostname_per_name_server:
        for hostname in hostnames:
            try:
                main_org = main_org_per_hostname[hostname]
            except KeyError:
                continue

            len_bgp_prefixes = len(bgp_prefixes_per_hostname[hostname])

            try:
                hostname_per_org_per_name_servers[name_server][main_org].append(
                    (hostname, len_bgp_prefixes)
                )
            except KeyError:
                hostname_per_org_per_name_servers[name_server][main_org] = [
                    (hostname, len_bgp_prefixes)
                ]

    # sort
    for name_server, hostname_per_org in hostname_per_org_per_name_servers.items():
        for org, hostname_bgp_prefixes in hostname_per_org.items():
            hostname_per_org_per_name_servers[name_server][org] = sorted(
                hostname_bgp_prefixes, key=lambda x: x[-1], reverse=True
            )

    return hostname_per_org_per_name_servers


def get_hostname_selection(hostname_per_org_per_name_servers: dict[dict]) -> dict[list]:
    """select hostname per organization/name servers"""
    selected_hostnames = defaultdict(dict)
    for name_server, hostname_main_org in hostname_per_org_per_name_servers.items():
        for main_org, hostnames in hostname_main_org.items():
            # TODO: vary this as well?
            for hostname, _ in hostnames[:10]:
                try:
                    selected_hostnames[name_server][main_org].append(hostname)
                except KeyError:
                    selected_hostnames[name_server][main_org] = [hostname]

    return selected_hostnames


def select_hostname_per_org_per_ns(
    name_servers_per_hostname: dict,
    tlds: list[str],
    bgp_per_cdn_per_hostname: dict,
    bgp_prefixes_per_hostname: dict,
    main_org_threshold: float = 0.2,
    bgp_prefix_threshold: int = 2,
) -> dict:
    """perform hostname selection per name server and organization"""
    name_servers_per_hostname = parse_name_servers(name_servers_per_hostname, tlds)

    hostname_per_name_servers = get_hostname_per_name_server(name_servers_per_hostname)

    hostname_per_main_org = get_hostname_per_main_org(
        bgp_per_cdn_per_hostname, main_org_threshold
    )

    hostname_per_main_org = parse_hostname_per_main_org(
        hostname_per_main_org, bgp_prefixes_per_hostname, bgp_prefix_threshold
    )

    main_org_per_hostname = {}
    for org, hostnames in hostname_per_main_org.items():
        for hostname in hostnames:
            main_org_per_hostname[hostname] = org

    hostname_per_org_per_name_servers = get_hostname_per_org_per_ns(
        hostname_per_name_servers, main_org_per_hostname, bgp_prefixes_per_hostname
    )
    selected_hostnames = get_hostname_selection(hostname_per_org_per_name_servers)

    return selected_hostnames


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
        for hostname, _ in bgp_prefix_per_hostname:

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
        for bgp_prefixes in bgp_prefixes_per_cdn.values():
            bgp_prefix_per_hostname[hostname].update(bgp_prefixes)

    hostname_per_cdn = defaultdict(dict)
    for hostname in cdn_per_hostname:
        for org, bgp_prefixes in cdn_per_hostname[hostname].items():
            hostname_per_cdn[org][hostname] = bgp_prefixes

    # hostname_per_cdn_greedy_bgp = select_hostname_greedy_bgp(
    #     cdn_per_hostname,
    #     bgp_prefix_per_hostname,
    # )
    # hostname_per_cdn_greedy_per_cdn = select_hostname_greedy_per_cdn(hostname_per_cdn)

    # hostname_per_cdn_max_bgp_prefix = select_hostname_max_bgp_prefix_per_cdn(
    #     hostname_per_cdn, bgp_prefix_per_hostname
    # )

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

    # dump_json(
    #     data=hostname_per_cdn_max_bgp_prefix,
    #     output_file=path_settings.DATASET / "hostname_per_cdn_max_bgp_prefix.json",
    # )

    name_servers_per_hostname = get_all_name_servers()
    tlds = load_csv(path_settings.DATASET / "tlds.csv")
    tlfs = [t.lower() for t in tlds]

    main_org_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    bgp_prefixes_thresholds = [2, 5, 10, 20, 50, 1_00, 150, 300, 5_00]

    hostname_selection = defaultdict(dict)
    for main_org_threshold in main_org_thresholds:
        for bgp_prefixes_threshold in bgp_prefixes_thresholds:
            selected_hostnames = select_hostname_per_org_per_ns(
                name_servers_per_hostname,
                tlds,
                cdn_per_hostname,
                bgp_prefix_per_hostname,
                main_org_threshold,
                bgp_prefixes_threshold,
            )

            nb_ns = selected_hostnames
            nb_org = set()
            nb_hostnames = set()
            for _, hostname_per_org in selected_hostnames.items():
                for org, hostnames in hostname_per_org.items():
                    nb_org.add(org)
                    nb_hostnames.update(hostnames)

            logger.info(f"{main_org_threshold=}")
            logger.info(f"{bgp_prefixes_threshold=}")
            logger.info(f"{len(nb_ns)=}")
            logger.info(f"{len(nb_org)=}")
            logger.info(f"{len(nb_hostnames)=}")
            logger.info("################################################")

            hostname_selection[main_org_threshold][
                bgp_prefixes_threshold
            ] = selected_hostnames

    dump_json(
        data=hostname_selection,
        output_file=path_settings.DATASET / "hostname_per_org_per_ns.json",
    )


if __name__ == "__main__":
    main()
