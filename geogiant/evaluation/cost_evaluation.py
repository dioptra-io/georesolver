from loguru import logger

from geogiant.common.queries import get_pings_per_target, load_vps, load_targets
from geogiant.common.files_utils import load_json
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

# load imc baseline evaluation
# naive approach len(vps * nb IP addresses)
# our methodo -> 50 VPs *
# RIPE IP map -> get from table?

if __name__ == "__main__":
    vps = load_vps(clickhouse_settings.VPS_FILTERED)
    targets = load_targets(clickhouse_settings.VPS_FILTERED)

    imc_baseline_results = load_json(
        path_settings.RESULTS_PATH / "round_based_algo_file.json"
    )

    imc_cost = sum([c[1] for c in imc_baseline_results[500]])

    logger.info("Measurement cost:: ")
    logger.info(f"All VPs: {len(vps) * len(targets)}")
    logger.info(f"IMC Baseline: {imc_cost}")
