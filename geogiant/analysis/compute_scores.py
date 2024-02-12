import asyncio

from numpy import mean
from tqdm import tqdm
from collections import defaultdict
from loguru import logger
from pych_client import AsyncClickHouseClient

from geogiant.clickhouse import (
    OverallScore,
    GetDNSMappingHostnames,
)

from geogiant.common.files_utils import dump_pickle, load_json, load_csv
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


async def get_overall_score(
    target_subnet: list,
    column_name: str,
    hostname_filter: tuple[str],
) -> None:
    subnet_score = {}
    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:
        resp = await OverallScore().execute(
            client=client,
            table_name="test_clustering",
            target_filter=target_subnet,
            column_name=column_name,
            hostname_filter=hostname_filter,
        )

    for row in resp:
        subnet_score[row["target_subnet"]] = row["scores"]

    return subnet_score


async def get_hostname_score(
    target_subnet: list,
    column_name: str,
    hostname_filter: tuple[str],
) -> None:
    vps = load_json(path_settings.OLD_VPS)
    targets = load_json(path_settings.OLD_TARGETS)
    vps_subnet = [get_prefix_from_ip(vp["address_v4"]) for vp in vps]
    targets_subnet = [get_prefix_from_ip(t["address_v4"]) for t in targets]

    async with AsyncClickHouseClient(**clickhouse_settings.clickhouse) as client:

        # get vps mapping
        resp = await GetDNSMappingHostnames().execute_iter(
            client=client,
            table_name="test_clustering",
            column_name=column_name,
            subnet_filter=[s for s in vps_subnet],
            hostname_filter=hostname_filter,
        )

        vps_mapping = defaultdict(dict)
        async for row in resp:
            subnet = row["subnet"]
            hostname = row["hostname"]
            mapping = row["mapping"]

            vps_mapping[subnet][hostname] = mapping

        # get targets mapping
        resp = await GetDNSMappingHostnames().execute_iter(
            client=client,
            table_name="test_clustering",
            column_name=column_name,
            subnet_filter=[s for s in targets_subnet],
            hostname_filter=hostname_filter,
        )

        targets_mapping = defaultdict(dict)
        async for row in resp:
            subnet = row["subnet"]
            hostname = row["hostname"]
            mapping = row["mapping"]

            targets_mapping[subnet][hostname] = mapping

        # compute scores
        target_scores = {}
        for target in tqdm(targets):
            target_subnet = get_prefix_from_ip(target["address_v4"])

            vps_score_per_hostname = defaultdict(list)
            for hostname, target_mapping in targets_mapping[target_subnet].items():
                for vp_subnet in vps_mapping:
                    try:
                        vp_mapping = vps_mapping[vp_subnet][hostname]
                    except KeyError:
                        logger.error(
                            f"Hostname::{hostname} not found for vp subnet = {vp_subnet}"
                        )

                    # TODO: check for mapping continental disparity (distance matrix?)
                    vps_score_per_hostname[vp_subnet].append(
                        len(set(target_mapping).intersection(set(vp_mapping)))
                        / min((len(set(target_mapping)), len(set(vp_mapping))))
                    )

            # get the avg number of match across hostnames as mapping score
            target_scores[target_subnet] = sorted(
                [
                    (vp_subnet, mean(hostname_scores))
                    for vp_subnet, hostname_scores in vps_score_per_hostname.items()
                ],
                key=lambda x: x[-1],
                reverse=True,
            )

    return target_scores


async def main() -> None:
    targets = load_json(path_settings.OLD_TARGETS)
    target_subnet = list(
        set([get_prefix_from_ip(target["address_v4"]) for target in targets])
    )
    hostname_filter = load_csv(path_settings.DATASET / "valid_hostnames_cdn.csv")

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: ANSWERS                   #")
    logger.info("#############################################")
    answers_score = await get_overall_score(target_subnet, "answer", hostname_filter="")
    dump_pickle(
        data=answers_score,
        output_file=path_settings.RESULTS_PATH
        / f"unfiltered_overall_answers_score.pickle",
    )

    answers_score = await get_overall_score(target_subnet, "answer", hostname_filter)
    dump_pickle(
        data=answers_score,
        output_file=path_settings.RESULTS_PATH / "overall_answers_score.pickle",
    )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: SUBNETS                   #")
    logger.info("#############################################")
    subnet_score = await get_overall_score(
        target_subnet, "answer_subnet", hostname_filter=""
    )
    dump_pickle(
        data=subnet_score,
        output_file=path_settings.RESULTS_PATH
        / f"unfiltered_overall_subnet_score.pickle",
    )

    subnet_score = await get_overall_score(
        target_subnet, "answer_subnet", hostname_filter
    )
    dump_pickle(
        data=subnet_score,
        output_file=path_settings.RESULTS_PATH / "overall_subnet_score.pickle",
    )

    logger.info("#############################################")
    logger.info("# OVERALL SCORE:: BGP PREFIXES              #")
    logger.info("#############################################")
    bgp_prefix_score = await get_overall_score(
        target_subnet, "answer_bgp_prefix", hostname_filter=""
    )
    dump_pickle(
        data=bgp_prefix_score,
        output_file=path_settings.RESULTS_PATH
        / f"unfiltered_overall_bgp_prefix_score.pickle",
    )

    bgp_prefix_score = await get_overall_score(
        target_subnet, "answer_bgp_prefix", hostname_filter
    )
    dump_pickle(
        data=bgp_prefix_score,
        output_file=path_settings.RESULTS_PATH / "overall_bgp_prefix_score.pickle",
    )

    # logger.info("#############################################")
    # logger.info("# OVERALL SCORE:: POP ID                    #")
    # logger.info("#############################################")
    # pop_id_score = await get_overall_score(target_subnet, "pop_ip_info_id", hostname_filter)
    # dump_pickle(
    #     data=pop_id_score,
    #     output_file=path_settings.RESULTS_PATH / "pop_id_score.pickle",
    # )


if __name__ == "__main__":
    asyncio.run(main())
