from collections import defaultdict, OrderedDict
from loguru import logger

from geogiant.common.files_utils import load_json, load_csv
from geogiant.evaluation.plot import ecdf, plot_cdf
from geogiant.common.settings import PathSettings

path_settings = PathSettings()

if __name__ == "__main__":

    best_hostnames_per_org_per_ns = load_json(
        path_settings.DATASET
        / "hostname_geo_score_selection_20_BGP_3_hostnames_per_org_ns.json",
    )

    cdn_per_hostname = load_json(
        path_settings.DATASET / "ecs_hostnames_organization.json"
    )

    bgp_prefix_per_hostname = defaultdict(set)
    for hostname, bgp_prefixes_per_cdn in cdn_per_hostname.items():
        for bgp_prefixes in bgp_prefixes_per_cdn.values():
            bgp_prefix_per_hostname[hostname].update(bgp_prefixes)

    bgp_prefixes_per_hostname = {}
    hostnames_len_bgp_prefixes = []

    nb_hostnames = set()
    for ns in best_hostnames_per_org_per_ns:
        for org, hostnames in best_hostnames_per_org_per_ns[ns].items():
            for hostname in hostnames:
                len_bgp_prefixes = len(bgp_prefix_per_hostname[hostname])

                bgp_prefixes_per_hostname[hostname] = (len_bgp_prefixes, org)
                hostnames_len_bgp_prefixes.append(len_bgp_prefixes)
                nb_hostnames.add(hostname)

    bgp_prefixes_per_hostname = sorted(
        bgp_prefixes_per_hostname.items(), key=lambda x: x[1][0], reverse=True
    )

    i = 0
    for hostname, (len_bgp_prefixes, org) in bgp_prefixes_per_hostname:
        print(f"{hostname}, {len_bgp_prefixes=}, {org=}")

        i += 1
        if i > 10:
            break

    logger.info(f"{len(nb_hostnames)} total hostnames")
    logger.info(f"{len(hostnames_len_bgp_prefixes)=}")
    x, y = ecdf(hostnames_len_bgp_prefixes)

    plot_cdf(
        x=x,
        y=y,
        output_path="hostnames_bgp_prefix_len_cdf",
        x_label="Number of BGP prefixes",
        y_label="Proportion of hostnames",
        x_lim=10,
    )
