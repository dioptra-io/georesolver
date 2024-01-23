import asyncio

from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import OverallScore
from geogiant.common.files_utils import dump_pickle, load_json
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def get_score(target_subnet: list, column_name: str) -> None:
    subnet_score = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await OverallScore().execute(
            client=client,
            table_name=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
            target_filter=target_subnet,
            column_name=column_name,
            hostname_filter=(
                "outlook.live.com",
                "docs.edgecast.com",
                "advancedhosting.com",
                "tencentcloud.com",
                "teams.microsoft.com",
                "cachefly.com",
            ),
        )

    for row in resp:
        subnet_score[row["target_subnet"]] = row["scores"]

    return subnet_score


async def main() -> None:
    targets = load_json(path_settings.OLD_TARGETS)
    target_subnet = list(
        set([get_prefix_from_ip(target["address_v4"]) for target in targets])
    )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: ANSWERS                   #")
    logger.info("#############################################")
    answers_score = await get_score(target_subnet, "answer")
    dump_pickle(
        data=answers_score,
        output_file=path_settings.RESULTS_PATH / "answers_score.pickle",
    )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: SUBNETS                   #")
    logger.info("#############################################")
    subnet_score = await get_score(target_subnet, "answer_subnet")
    dump_pickle(
        data=subnet_score,
        output_file=path_settings.RESULTS_PATH / "subnet_score.pickle",
    )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: BGP PREFIXES              #")
    logger.info("#############################################")
    bgp_prefix_score = await get_score(target_subnet, "answer_bgp_prefix")
    dump_pickle(
        data=bgp_prefix_score,
        output_file=path_settings.RESULTS_PATH / "bgp_prefix_score.pickle",
    )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: POP ID                    #")
    logger.info("#############################################")
    pop_id_score = await get_score(target_subnet, "pop_ip_info_id")
    dump_pickle(
        data=pop_id_score,
        output_file=path_settings.RESULTS_PATH / "pop_id_score.pickle",
    )


if __name__ == "__main__":
    asyncio.run(main())
