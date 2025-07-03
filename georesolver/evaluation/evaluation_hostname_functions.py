import os
import asyncio
import numpy as np

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from loguru import logger
from pych_client import ClickHouseClient

from georesolver.clickhouse import GetAllNameServers
from georesolver.evaluation.evaluation_plot_functions import plot_cdf, ecdf
from georesolver.common.files_utils import (
    dump_csv,
    load_csv,
    load_json,
    dump_json,
)
from georesolver.clickhouse.queries import get_mapping_per_hostname
from georesolver.common.ip_addresses_utils import get_prefix_from_ip
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_all_name_servers(name_servers_table: str) -> dict:
    """retrieve all unparsed name servers"""
    name_servers_per_hostname = {}
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetAllNameServers().execute(client, name_servers_table)

        for row in rows:
            name_servers_per_hostname[row["hostname"]] = row["name_servers"]

    return name_servers_per_hostname


def parse_name_servers(name_servers_per_hostname: dict, tlds: list[str]) -> dict:
    """parse name server to extract main organization behin them"""
    tlds = [tld for tld in tlds if len(tld) < 3]

    parsed_name_servers_per_hostname = {}
    for hostname, name_servers in name_servers_per_hostname.items():
        parsed_name_servers = set()
        for name_server in name_servers:

            test_name_server = name_server
            name_server = name_server.split(".")

            # remove tlds belonging to country
            for tld in tlds:
                if len(tld) < 3:
                    try:
                        name_server = list(filter((tld).__ne__, name_server))
                    except ValueError:
                        continue

            # remove all name servers belonging to state
            if "gov" in name_server or "edu" in name_server:
                continue

            other_tlds = [
                "com",
                "net",
                "org",
                "COM",
                "gov",
                "edu",
                "pro",
                "go",
                "one",
                "law",
                "top",
                "nyc",
                "icu",
                "nsw",
                "biz",
                "dev",
                "mi",
                "pl2",
                "bac",
                "cloud",
            ]
            for tld in other_tlds:
                try:
                    name_server = list(filter((tld).__ne__, name_server))
                except ValueError:
                    continue

            # remove whitespace
            try:
                name_server.remove("")
            except ValueError:
                pass

            if len(name_server) == 1:
                parsed_name_server = name_server[0]
            else:
                parsed_name_server = name_server[-1]

            if "awsdns" in parsed_name_server:
                parsed_name_server = "awsdns"

            if "msedge" in parsed_name_server:
                parsed_name_server = "msedge"

            if "bunny" in parsed_name_server:
                parsed_name_server = "bunny"

            if "akam" in parsed_name_server or "akamai-edge" in parsed_name_server:
                parsed_name_server = "akamai"

            if "incap" in parsed_name_server:
                parsed_name_server = "impervadns"

            if "alidns" in parsed_name_server:
                parsed_name_server = "alibabadns"

            # manually remove ns identified as wrong
            if parsed_name_server in [
                "customer",
                "invalid",
                "gslb",
                "president",
                "ns",
                "mail",
                "no-ip",
                "dns",
                "gob",
                "ok",
                "fun",
                "ha",
                "z",
                "wni",
                "rdw",
                "meh",
                "aon",
                "cmp",
            ]:
                continue

            try:
                int(parsed_name_server)
                continue
            except ValueError:
                pass

            if len(parsed_name_server) <= 3:
                # remove occurence of ns1, ns2, etc.
                if "ns" in parsed_name_server:
                    continue

                # small lenght name servers that are valid
                if parsed_name_server not in [
                    "qq",
                    "ibm",
                    "ovh",
                    "wp",
                    "irs",
                    "nic",
                    "hoy",
                    "cgi",
                    "kdg",
                    "uia",
                    "ps",
                    "ngi",
                    "sdv",
                    "nos",
                    "ati",
                    "t2v",
                    "uptr",
                    "ccb",
                    "rik",
                    "ahn",
                    "tcs",
                    "ps",
                    "sk",
                    "ccb",
                ]:
                    continue

            parsed_name_servers.add(parsed_name_server)

        parsed_name_servers_per_hostname[hostname] = parsed_name_servers

    return parsed_name_servers_per_hostname


def get_hostname_per_name_server(parsed_name_servers: dict) -> list:
    """get all hostnames that have a unique name server"""
    hostname_per_name_server = defaultdict(list)
    filtered_hostnames = set()
    remaining_hostnames = set()
    for hostname, name_server in parsed_name_servers.items():

        if len(name_server) != 1:
            filtered_hostnames.add(hostname)
            continue
        else:
            remaining_hostnames.add(hostname)
            name_server = list(name_server)[0]

        hostname_per_name_server[name_server].append(hostname)

    hostname_per_name_server = sorted(
        hostname_per_name_server.items(), key=lambda x: len(x[-1]), reverse=True
    )

    logger.info(f"Original number of hostnames:: {len(parsed_name_servers)}")
    logger.info(f"Nb of hostname rejected:: {len(filtered_hostnames)}")
    logger.info(f"Nb of hostname remaining:: {len(remaining_hostnames)}")
    logger.info(f"Number of name servers:: {len(hostname_per_name_server)}")

    # filter out name servers for which we only have one hostname
    filtered_hostname_per_name_server = []
    filtered_hostnames = set()
    for name_server, hostnames in hostname_per_name_server:
        if len(hostnames) > 1:
            filtered_hostname_per_name_server.append((name_server, hostnames))

        else:
            filtered_hostnames.update(hostnames)

    logger.info(
        f"Remaining name servers after fitering minor:: {len(filtered_hostname_per_name_server)}"
    )
    logger.info(f"Filtered hostnames:: {len(filtered_hostnames)}")

    return filtered_hostname_per_name_server


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


def merge_main_orgs(main_org: str) -> None:
    """merge main organization (ex: AMAZON-02 and AMAZON-AES) -> AMAZON"""
    if "AMAZON" in main_org:
        main_org = "AMAZON"

    elif "GOOGLE" in main_org:
        main_org = "GOOGLE"

    elif "AKAMAI" in main_org:
        main_org = "AKAMAI"

    elif "APPLE" in main_org:
        main_org = "APPLE"

    elif "MICROSOFT" in main_org:
        main_org = "MICROSOFT"

    elif "TENCENT" in main_org:
        main_org = "TENCENT"

    elif "CHINANET" in main_org:
        main_org = "CHINANET"

    elif "CMNET" in main_org:
        main_org = "CMNET"

    else:
        main_org = main_org

    return main_org


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
    name_server_per_hostnames = {}
    for name_server, hostname_main_org in hostname_per_org_per_name_servers.items():
        for main_org, hostnames in hostname_main_org.items():
            for hostname, _ in hostnames:
                try:
                    selected_hostnames[name_server][main_org].append(hostname)
                    name_server_per_hostnames[hostname] = name_server

                except KeyError:
                    selected_hostnames[name_server][main_org] = [hostname]
                    name_server_per_hostnames[hostname] = name_server

    return selected_hostnames, name_server_per_hostnames


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
    selected_hostnames, name_server_per_hostnames = get_hostname_selection(
        hostname_per_org_per_name_servers
    )

    return selected_hostnames, name_server_per_hostnames


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
        for hostname, bgp_prefixes_in_org in bgp_prefix_per_hostname:

            # some hostnames has more than just one org
            hostname_bgp_prefixes = bgp_prefixes_per_hostname[hostname]

            if len(hostname_bgp_prefixes) > 10:
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


def get_hostname_geo_score(nb_vps: int, bgp_prefix_per_hostname: dict[list]) -> dict:
    """
    get all vps mapping, take N vps that maximize geo distance
    compute redirection similarity over the N vps
    if high similarity, hostname does not provide usefull geo info
    """
    # get hostnames geo score (inverse of georesolver)
    hostname_geo_score_path = path_settings.RESULTS_PATH / "hostname_geo_score.json"
    if (hostname_geo_score_path).exists():
        return load_json(hostname_geo_score_path)

    hostnames = [
        h for h, bgp_pref in bgp_prefix_per_hostname.items() if len(bgp_pref) > 1
    ]

    logger.info(f"{len(hostnames)} hostnames with more than one BGP prefix returned")

    greedy_vps = load_json(path_settings.GREEDY_VPS)
    greedy_vps = [get_prefix_from_ip(ip) for ip in greedy_vps[:nb_vps]]

    mapping_per_hostname = get_mapping_per_hostname(
        dns_table="vps_ecs_mapping",
        subnets=[s for s in greedy_vps],
        hostname_filter="" if not hostnames else [h for h in hostnames],
    )
    hostname_similarity_score = {}
    for hostname, vps_mapping in tqdm(mapping_per_hostname.items()):
        pairwise_similarity = []
        for i, vp_i in enumerate(greedy_vps):
            try:
                mapping_i = vps_mapping[vp_i]
            except KeyError:
                continue

            for j, vp_j in enumerate(greedy_vps):
                if vp_i == vp_j:
                    continue

                try:
                    mapping_j = vps_mapping[vp_j]
                except KeyError:
                    continue

                jaccard_index = len(set(mapping_i).intersection(set(mapping_j))) / len(
                    set(mapping_i).union(set(mapping_j))
                )

                pairwise_similarity.append(jaccard_index)

        # take average pairwise similarity
        if pairwise_similarity:
            hostname_similarity_score[hostname] = np.mean(pairwise_similarity)

    hostname_similarity_score = sorted(
        hostname_similarity_score.items(), key=lambda x: x[-1]
    )

    dump_json(hostname_similarity_score, hostname_geo_score_path)

    return hostname_similarity_score


def get_bpg_prefix_per_hostnames(cdn_per_hostname: dict) -> dict:
    """get all bgp prefixes per hostnames,
    filter hostnames for which we only have one bgp prefix
    """
    bgp_prefix_per_hostname = defaultdict(set)
    for hostname, bgp_prefixes_per_cdn in cdn_per_hostname.items():
        for bgp_prefixes in bgp_prefixes_per_cdn.values():
            bgp_prefix_per_hostname[hostname].update(bgp_prefixes)

    count = 0
    count_1 = 0
    for hostname, bgp_prefixes in bgp_prefix_per_hostname.items():
        if len(bgp_prefixes) < 10:
            count += 1
        if not len(bgp_prefixes) > 1:
            count_1 += 1

    logger.info(
        f"Removed:: {count}/{len(bgp_prefix_per_hostname)}/{round(count * 100 / len(bgp_prefix_per_hostname),2)} hostname with less than 10"
    )

    logger.info(
        f"Removed:: {count_1}/{len(bgp_prefix_per_hostname)}/{round(count_1 * 100 / len(bgp_prefix_per_hostname),2)} hostname with less than 1"
    )

    return bgp_prefix_per_hostname


def get_best_hostnames_per_org(
    hostname_geo_score: list,
    cdn_per_hostname: dict,
    bgp_prefix_per_hostnames: dict,
    ns_per_hostname: dict,
    hostnames_per_org_path: Path = (
        path_settings.DATASET / "best_hostnames_per_org_new.json"
    ),
    hostnames_per_org_per_ns_path: Path = (
        path_settings.DATASET / "best_hostnames_per_org_per_ns_new.json"
    ),
) -> tuple[dict, dict]:
    """for each organization get the best set of hostnames"""

    # get the best hostnames per organization
    if hostnames_per_org_path.exists():
        best_hostnames_per_org = load_json(hostnames_per_org_path)
    else:
        best_hostnames_per_org = defaultdict(list)
        for index, (hostname, score) in enumerate(hostname_geo_score):
            bgp_prefixes_per_org = cdn_per_hostname[hostname]
            main_org, _ = max(bgp_prefixes_per_org.items(), key=lambda x: len(x[-1]))

            main_org = merge_main_orgs(main_org)
            hostname_bgp_prefixes = bgp_prefix_per_hostnames[hostname]

            best_hostnames_per_org[main_org].append(
                (index, hostname, score, len(hostname_bgp_prefixes))
            )

        dump_json(best_hostnames_per_org, hostnames_per_org_path)

    if hostnames_per_org_per_ns_path.exists():
        best_hostnames_per_org_per_ns = load_json(hostnames_per_org_per_ns_path)
    else:
        # get the best hostnames per organization AND per NS
        best_hostnames_per_org_per_ns = defaultdict(dict)
        for org, hostnames in best_hostnames_per_org.items():
            for i, hostname, score, nb_hostnames_bgp_prefixes in hostnames:
                try:
                    name_server = ns_per_hostname[hostname]
                except KeyError:
                    continue

                try:
                    best_hostnames_per_org_per_ns[name_server][org].append(
                        (i, hostname, score, nb_hostnames_bgp_prefixes)
                    )
                except KeyError:
                    best_hostnames_per_org_per_ns[name_server][org] = [
                        (i, hostname, score, nb_hostnames_bgp_prefixes)
                    ]

        dump_json(best_hostnames_per_org_per_ns, hostnames_per_org_per_ns_path)

    return best_hostnames_per_org, best_hostnames_per_org_per_ns


def get_hostnames_org_and_ns_threshold(
    best_hostnames_per_org_per_ns: dict, output_file_root: str = "hostname_georesolver_"
) -> None:
    """
    generate dataset of hostnames function of the number of org/ns
    threhold and bgp prefixes threshold parameters
    """
    ns_org_thresholds = [3, 5, 10]
    bgp_thresholds = [10, 20, 50]

    for ns_org_threhsold in ns_org_thresholds:
        for bgp_threshold in bgp_thresholds:
            filtered_hostname_per_org = defaultdict(dict)
            for ns, hostnames_per_org in best_hostnames_per_org_per_ns.items():
                for org, hostnames in hostnames_per_org.items():

                    hostnames = sorted(hostnames, key=lambda x: x[2])

                    for index, hostname, score, nb_bgp_prefixes in hostnames:

                        # filter hostnames with less than 10 BGP prefixes
                        if nb_bgp_prefixes < bgp_threshold:
                            continue

                        try:
                            if (
                                len(filtered_hostname_per_org[ns][org])
                                < ns_org_threhsold
                            ):

                                filtered_hostname_per_org[ns][org].append(hostname)
                        except KeyError:
                            filtered_hostname_per_org[ns][org] = [hostname]

            total_hostnames = set()
            selected_hostnames = {}
            nb_pairs = set()
            nb_orgs = set()
            for ns in filtered_hostname_per_org:
                # logger.info(f"{ns}")
                for org, hostnames_score in filtered_hostname_per_org[ns].items():
                    # logger.info(f"{org=}, {len(hostnames_score)}")
                    total_hostnames.update([hostname for hostname in hostnames_score])
                    selected_hostnames[org] = [hostname for hostname in hostnames_score]
                    nb_orgs.add(org)
                    nb_pairs.add((ns, org))

            logger.info(
                f"Per NS/org:: {bgp_threshold=}, {ns_org_threhsold=} {len(total_hostnames)=}"
            )
            logger.info(f"Nb Name servers:: {len(filtered_hostname_per_org)}")
            logger.info(f"Nb orgs:: {len(nb_orgs)}")
            logger.info(f"Nb pairs:: {len(nb_pairs)}")
            logger.info(f"Total hostnames:: {len(total_hostnames)}")

            dump_json(
                filtered_hostname_per_org,
                path_settings.DATASET
                / f"hostname_{bgp_threshold}_BGP_{ns_org_threhsold}_org_ns_new.json",
            )

            dump_csv(
                total_hostnames,
                path_settings.HOSTNAME_FILES
                / f"{output_file_root}_{bgp_threshold}_BGP_{ns_org_threhsold}_org_ns_new.csv",
            )


async def main() -> None:
    """
    based on ECS resolution on RIPE Atlas VPs, generate CSV files following different methodology:
    1) select greedy bgp prefixes (maximize coverage of bgp prefix)
    2) select greedy bgp prefixes per CDN (same but maximize bgp prefix coverage PER CDN)
    3) select hostnames that returned the maximum number of BGP prefixes, per CDN
    """
    cdn_per_hostname = load_json(
        path_settings.HOSTNAME_FILES / "ecs_hostnames_organization.json"
    )
    bgp_prefix_per_hostnames = get_bpg_prefix_per_hostnames(cdn_per_hostname)
    name_servers_per_hostname = get_all_name_servers("hostname_name_servers")
    missing_name_servers_per_hostname = get_all_name_servers(
        "hostname_name_servers_end_to_end"
    )
    name_servers_per_hostname.update(missing_name_servers_per_hostname)
    hostname_geo_score = get_hostname_geo_score(1_000, bgp_prefix_per_hostnames)

    # select hostnames based on their name server and organisation
    tlds = load_csv(path_settings.DATASET / "tlds.csv")
    tlds = [t.lower() for t in tlds]
    selected_hostnames, ns_per_hostname = select_hostname_per_org_per_ns(
        name_servers_per_hostname=name_servers_per_hostname,
        tlds=tlds,
        bgp_per_cdn_per_hostname=cdn_per_hostname,
        bgp_prefixes_per_hostname=bgp_prefix_per_hostnames,
        main_org_threshold=0.01,
        bgp_prefix_threshold=2,
    )

    # save files
    _, best_hostnames_per_org_per_ns = get_best_hostnames_per_org(
        hostname_geo_score=hostname_geo_score,
        cdn_per_hostname=cdn_per_hostname,
        bgp_prefix_per_hostnames=bgp_prefix_per_hostnames,
        ns_per_hostname=ns_per_hostname,
    )

    get_hostnames_org_and_ns_threshold(best_hostnames_per_org_per_ns)


if __name__ == "__main__":
    asyncio.run(main())
