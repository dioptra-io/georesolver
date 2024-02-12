import asyncio
import sys

from loguru import logger

from geogiant.vp_selection import VPSelectionDNS

from geogiant.common.files_utils import load_json, dump_pickle
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


async def main() -> None:
    """main fct script"""
    probing_budgets = [1]
    probing_budgets.extend([i for i in range(10, 100, 10)])
    probing_budgets.extend([i for i in range(100, 500, 100)])

    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    results = {}
    for probing_budget in probing_budgets:
        logger.info(f"ECS-DNS geolocation with {probing_budget} VPs")

        geoloc_errors, min_latencies = await VPSelectionDNS().main(
            targets=targets,
            vps=vps,
            output_path=f"ecs_dns_{probing_budget}",
            target_selection=True,
            subnet_selection=False,
            probing_budget=probing_budget,
        )

        results[probing_budget] = (geoloc_errors, min_latencies)

    dump_pickle(
        results, path_settings.RESULTS_PATH / "ecs_dns_function_of_prob_budget.pickle"
    )


if __name__ == "__main__":
    asyncio.run(main())
