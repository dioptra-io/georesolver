import asyncio

from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import OverallScore
from geogiant.common.files_utils import dump_pickle, load_json
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

hostname_filter = (
    "outlook.live.com",
    "docs.edgecast.com",
    "advancedhosting.com",
    "tencentcloud.com",
    "teams.microsoft.com",
    "news.yahoo.co.jp",
    "baseball.yahoo.co.jp",
    "www.yahoo.co.jp",
    "finance.yahoo.co.jp",
    "weather.yahoo.co.jp",
    "cachefly.com",
    "fastly.com",
    "www.google.co.th",
    "www.google.com.tr",
    "forms.office.com",
    "detail.chiebukuro.yahoo.co.jp",
    "sports.yahoo.co.jp",
    "weather.yahoo.co.jp",
    "auctions.yahoo.co.jp",
    "page.auctions.yahoo.co.jp",
    "search.yahoo.co.jp",
    "chrome.google.com",
    "calendar.google.com",
    "business.google.com",
    "classroom.google.com",
    "www.facebook.com",
    "www.instagram.com",
    "accounts.google.com",
    "docs.google.com",
    "drive.google.com",
    "mail.google.com",
    "meet.google.com",
    "news.google.com",
    "one.google.com",
    "photos.google.com",
    "scholar.google.com",
    "sites.google.com",
    "studio.youtube.com",
    "www.google.pl",
    "www.office.com",
    "support.google.com",
    "www.google.ca",
    "www.google.cl",
    "www.google.co.in",
    "www.google.co.jp",
    "www.google.com.ar",
    "www.google.co.uk",
    "www.google.com.tw",
    "www.google.de",
    "www.google.fr",
    "www.google.it",
    "www.google.nl",
    "www.google.p",
    "myactivity.google.com",
    "translate.google.com",
    "myaccount.google.com",
    "play.google.com",
    "www.google.com.br",
    "www.google.es",
    "www.google.com.mx",
    "www.google.com.hk",
)

hostname_filter = (
    "outlook.live.com",
    "docs.edgecast.com",
    "www.office.com",
    "advancedhosting.com",
    "tencentcloud.com",
    "teams.microsoft.com",
    "news.yahoo.co.jp",
    "baseball.yahoo.co.jp",
    "www.yahoo.co.jp",
    "finance.yahoo.co.jp",
    "weather.yahoo.co.jp",
    "cachefly.com",
    "fastly.com",
    "forms.office.com",
    "detail.chiebukuro.yahoo.co.jp",
    "sports.yahoo.co.jp",
    "weather.yahoo.co.jp",
    "auctions.yahoo.co.jp",
    "page.auctions.yahoo.co.jp",
    "search.yahoo.co.jp",
    "www.facebook.com",
    "www.instagram.com",
    "www.google.pl",
    "www.google.ca",
    "www.google.cl",
    "www.google.co.in",
    "www.google.co.jp",
    "www.google.com.ar",
    "www.google.co.uk",
    "www.google.com.tw",
    "www.google.de",
    "www.google.fr",
    "www.google.it",
    "www.google.nl",
    "www.google.p",
    "www.google.com.br",
    "www.google.es",
    "www.google.com.mx",
    "www.google.com.hk",
    "www.google.co.th",
    "www.google.com.tr",
)


async def get_score(
    target_subnet: list,
    column_name: str,
    hostname_filter: tuple[str],
) -> None:
    subnet_score = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await OverallScore().execute(
            client=client,
            table_name=clickhouse_settings.OLD_DNS_MAPPING_WITH_METADATA,
            target_filter=target_subnet,
            column_name=column_name,
            hostname_filter=hostname_filter,
        )

    for row in resp:
        subnet_score[row["target_subnet"]] = row["scores"]

    return subnet_score


async def main() -> None:
    targets = load_json(path_settings.OLD_TARGETS)
    target_subnet = list(
        set([get_prefix_from_ip(target["address_v4"]) for target in targets])
    )

    # logger.info("#############################################")
    # logger.info("# OVERALL SCORE:: ANSWERS                   #")
    # logger.info("#############################################")
    # answers_score = await get_score(target_subnet, "answer", hostname_filter="")
    # dump_pickle(
    #     data=answers_score,
    #     output_file=path_settings.RESULTS_PATH / "unfiltered_answers_score.pickle",
    # )

    # answers_score = await get_score(target_subnet, "answer", hostname_filter)
    # dump_pickle(
    #     data=answers_score,
    #     output_file=path_settings.RESULTS_PATH / "answers_score.pickle",
    # )

    # logger.info("#############################################")
    # logger.info("# OVERALL SCORE:: SUBNETS                   #")
    # logger.info("#############################################")
    # subnet_score = await get_score(target_subnet, "answer_subnet", hostname_filter="")
    # dump_pickle(
    #     data=subnet_score,
    #     output_file=path_settings.RESULTS_PATH / "unfiltered_subnet_score.pickle",
    # )

    # subnet_score = await get_score(target_subnet, "answer_subnet", hostname_filter)
    # dump_pickle(
    #     data=subnet_score,
    #     output_file=path_settings.RESULTS_PATH / "subnet_score.pickle",
    # )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: BGP PREFIXES              #")
    logger.info("#############################################")
    bgp_prefix_score = await get_score(
        target_subnet, "answer_bgp_prefix", hostname_filter=""
    )
    dump_pickle(
        data=bgp_prefix_score,
        output_file=path_settings.RESULTS_PATH / "unfiltered_bgp_prefix_score.pickle",
    )

    bgp_prefix_score = await get_score(
        target_subnet, "answer_bgp_prefix", hostname_filter=hostname_filter
    )
    dump_pickle(
        data=bgp_prefix_score,
        output_file=path_settings.RESULTS_PATH / "bgp_prefix_score.pickle",
    )

    # logger.info("#############################################")
    # logger.info("# OVERALL SCORE:: POP ID                    #")
    # logger.info("#############################################")
    # pop_id_score = await get_score(target_subnet, "pop_ip_info_id", hostname_filter)
    # dump_pickle(
    #     data=pop_id_score,
    #     output_file=path_settings.RESULTS_PATH / "pop_id_score.pickle",
    # )


if __name__ == "__main__":
    asyncio.run(main())
