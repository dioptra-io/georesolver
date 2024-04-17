from loguru import logger
from collections import defaultdict

from geogiant.common.files_utils import dump_pickle, load_json
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

if __name__ == "__main__":
    # hostname_file = "hostname_per_cdn_max_bgp_prefix.json"
    hostname_file = "hostname_per_org_per_ns.json"

    targets_table = clickhouse_settings.VPS_FILTERED
    vps_table = clickhouse_settings.VPS_FILTERED

    targets_ecs_table = "vps_mapping_ecs"
    vps_ecs_table = "vps_mapping_ecs"

    hostname_per_cdn_per_ns = load_json(path_settings.DATASET / hostname_file)

    selected_hostnames_per_cdn = defaultdict(list)

    main_org_thresholds = [0.2, 0.4, 0.6, 0.8]
    bgp_prefixes_thresholds = [2, 5, 10, 50, 1_00, 5_00]

    for main_org_threshold in main_org_thresholds:
        for bgp_prefixes_threshold in bgp_prefixes_thresholds:

            logger.info(
                f"Running score calculation with parameters:: {main_org_threshold=}, {bgp_prefixes_threshold=}"
            )

            # extract hostname per cdn
            for ns in hostname_per_cdn_per_ns[str(main_org_threshold)][
                str(bgp_prefixes_threshold)
            ]:
                for cdn, hostnames in hostname_per_cdn_per_ns[str(main_org_threshold)][
                    str(bgp_prefixes_threshold)
                ][ns].items():
                    selected_hostnames_per_cdn[cdn].extend(hostnames)
                    logger.info(
                        f"{cdn=}, nb_hostnames={len(selected_hostnames_per_cdn[cdn])}"
                    )

            score_config = {
                "targets_table": targets_table,
                "main_org_threshold": main_org_threshold,
                "bgp_prefixes_threshold": bgp_prefixes_threshold,
                "vps_table": vps_table,
                "hostname_per_cdn_per_ns": hostname_per_cdn_per_ns,
                "hostname_per_cdn": selected_hostnames_per_cdn,
                "targets_ecs_table": targets_ecs_table,
                "vps_ecs_table": vps_ecs_table,
                "hostname_selection": "max_bgp_prefix",
                "score_metric": [
                    # "intersection",
                    "jaccard",
                    # "jaccard_scope_linear_weight",
                    # "jaccard_scope_poly_weight",
                    # "jaccard_scope_exp_weight",
                    # "intersection_scope_linear_weight",
                    # "intersection_scope_poly_weight",
                    # "intersection_scope_exp_weight",
                ],
                "answer_granularities": [
                    # "answers",
                    # "answer_subnets",
                    "answer_bgp_prefixes",
                ],
            }

            output_path = (
                path_settings.RESULTS_PATH
                / f"scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_{main_org_threshold}_bgp_prefix_threshold_{bgp_prefixes_threshold}.pickle"
            )

            score = get_scores(score_config)

            dump_pickle(data=score, output_file=output_path)

            logger.info(f"Score calculation done, {output_path=}")
