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

    targets_ecs_table = "filtered_hostnames_ecs_mapping"
    vps_ecs_table = "filtered_hostnames_ecs_mapping"

    hostname_per_cdn = load_json(path_settings.DATASET / hostname_file)

    orgs = ["GOOGLE"]

    for nb_hostnames in [1, 5, 10, 50, 1_00, 5_00, 1_000]:

        selected_hostnames_per_cdn = {}
        for org in orgs:
            selected_hostnames_per_cdn[org] = hostname_per_cdn[org][:nb_hostnames]
            logger.info(f"{org=}, nb_hostnames={len(selected_hostnames_per_cdn[org])}")

        score_config = {
            "targets_table": targets_table,
            "vps_table": vps_table,
            "hostname_per_cdn": selected_hostnames_per_cdn,
            "targets_ecs_table": targets_ecs_table,
            "vps_ecs_table": vps_ecs_table,
            "hostname_selection": "max_bgp_prefixes",
            "score_metric": [
                "intersection",
                "jaccard",
                # "jaccard_scope_linear_weight",
                # "jaccard_scope_poly_weight",
                # "jaccard_scope_exp_weight",
                # "intersection_scope_linear_weight",
                # "intersection_scope_poly_weight",
                # "intersection_scope_exp_weight",
            ],
            "answer_granularities": ["answer_bgp_prefixes"],
        }

        output_path = (
            path_settings.RESULTS_PATH
            / f"scores__{org}_{len(selected_hostnames_per_cdn[org])}_{score_config['hostname_selection']}.pickle"
        )

        get_scores(score_config)

        logger.info(f"Score calculation done, {output_path=}")
