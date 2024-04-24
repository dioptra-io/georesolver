from loguru import logger

from geogiant.common.files_utils import dump_pickle, load_json
from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

if __name__ == "__main__":
    hostname_file = "hostname_per_cdn_max_bgp_prefix.json"

    targets_table = clickhouse_settings.VPS_FILTERED
    vps_table = clickhouse_settings.VPS_FILTERED

    targets_ecs_table = "vps_mapping_ecs"
    vps_ecs_table = "vps_mapping_ecs"

    hostname_per_cdn = load_json(path_settings.DATASET / hostname_file)

    nb_hostnames = 10
    selected_hostnames_per_cdn = {}
    for cdn, hostnames in hostname_per_cdn.items():
        selected_hostnames_per_cdn[cdn] = hostname_per_cdn[cdn][:nb_hostnames]
        logger.info(f"{cdn=}, nb_hostnames={len(selected_hostnames_per_cdn[cdn])}")

    output_path = (
        path_settings.RESULTS_PATH
        / f"scores__all_cdns_10_hostname_per_cdn_max_bgp_prefix.pickle"
    )

    score_config = {
        "targets_table": targets_table,
        "vps_table": vps_table,
        "hostname_per_cdn": selected_hostnames_per_cdn,
        "targets_ecs_table": targets_ecs_table,
        "vps_ecs_table": vps_ecs_table,
        "hostname_selection": "max_bgp_prefix",
        "score_metric": [
            # "intersection",
            "jaccard",
            "jaccard_scope_linear_weight",
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
        "output_path": output_path,
    }

    get_scores(score_config)

    logger.info(f"Score calculation done, {output_path=}")
