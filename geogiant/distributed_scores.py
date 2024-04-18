import subprocess

from tqdm import tqdm
from loguru import logger
from fabric import Connection
from collections import defaultdict
from pathlib import Path
from pych_client import ClickHouseClient

from geogiant.clickhouse import (
    GetHostnames,
    GetAllDNSMapping,
    CreateDNSMappingTable,
    CreateNameServerTable,
    InsertFromCSV,
)
from geogiant.common.files_utils import (
    load_json,
    load_csv,
    dump_csv,
    dump_json,
    dump_pickle,
    create_tmp_csv_file,
    dump_json,
)

from geogiant.ecs_vp_selection.scores import get_scores
from geogiant.common.queries import (
    load_vp_subnets,
    load_target_subnets,
    get_subnets_mapping,
)
from geogiant.common.settings import ClickhouseSettings, PathSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def docker_run_cmd() -> str:
    return """
        docker run -d \
        -v "$(pwd)/results:/app/geogiant/results" \
        -v "$(pwd)/datasets/targets_subnet.json:/app/geogiant/datasets/targets_subnet.json" \
        -v "$(pwd)/datasets/vps_subnet.json:/app/geogiant/datasets/vps_subnet.json" \
        -v "$(pwd)/datasets/targets_mapping.pickle:/app/geogiant/datasets/targets_mapping.pickle" \
        -v "$(pwd)/datasets/vps_mapping.pickle:/app/geogiant/datasets/vps_mapping.pickle" \
        -v "$(pwd)/datasets/score_config.json:/app/geogiant/datasets/score_config.json" \
        ghcr.io/dioptra-io/geogiant:main
    """


def load_hostnames(
    hostname_per_cdn_per_ns: dict,
    main_org_threshold: float,
    bgp_prefixes_threshold: int,
) -> dict:
    selected_hostnames = set()
    selected_hostnames_per_cdn = defaultdict(list)
    for ns in hostname_per_cdn_per_ns[str(main_org_threshold)][
        str(bgp_prefixes_threshold)
    ]:
        for cdn, hostnames in hostname_per_cdn_per_ns[str(main_org_threshold)][
            str(bgp_prefixes_threshold)
        ][ns].items():
            selected_hostnames_per_cdn[cdn].extend(hostnames)
            selected_hostnames.update(hostnames)

    return selected_hostnames_per_cdn, list(selected_hostnames)


def deploy_score(
    vm: str,
    vm_config: dict,
) -> None:
    """run docker image on gcp VMs"""
    logger.info(f"Running ECS resolution on:: {vm}")

    score_config = vm_config["score_config"]

    c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")

    # create dataset and result dir
    result = c.run("mkdir -p results")
    result = c.run("mkdir -p datasets")

    vm_result_path = path_settings.RESULTS_PATH / f"{vm}"

    # create result dir for vm
    result = subprocess.run(
        f"mkdir -p {vm_result_path}", shell=True, capture_output=True, text=True
    )

    logger.info(f"Results ouput dir:: {vm_result_path}")

    # Dump score config to local vm folder
    vm_score_config_path = vm_result_path / "score_config.json"
    dump_json(score_config, vm_score_config_path)

    # upload score config file to dataset folder
    result = c.put(
        local=f"{vm_score_config_path}",
        remote="datasets",
    )

    # load target and vps subnets
    targets_subnet = load_json(Path(score_config["targets_subnet_path"]))
    vps_subnet = load_json(Path(score_config["vps_subnet_path"]))

    # get target and vps mapping
    targets_mapping = get_subnets_mapping(
        targets_ecs_table,
        subnets=[s for s in targets_subnet],
        hostname_filter=selected_hostnames,
    )

    vps_mapping = get_subnets_mapping(
        vps_ecs_table,
        subnets=[s for s in vps_subnet],
        hostname_filter=selected_hostnames,
    )

    # dump mapping files
    dump_pickle(targets_mapping, vm_result_path / "targets_mapping.pickle")
    dump_pickle(vps_mapping, vm_result_path / "vps_mapping.pickle")

    # upload mapping files
    result = c.put(
        local=f"{vm_result_path / 'targets_mapping.pickle'}",
        remote="datasets",
    )
    result = c.put(
        local=f"{vm_result_path / 'vps_mapping.pickle'}",
        remote="datasets",
    )

    # docker login
    result = c.run(f"export PAT={path_settings.GITHUB_TOKEN}")
    result = c.run(
        f"docker login ghcr.io -u {path_settings.DOCKER_USERNAME} -p {path_settings.GITHUB_TOKEN}"
    )

    # pull docker image
    result = c.run("docker pull ghcr.io/dioptra-io/geogiant:main")

    # run docker image
    # logger.info("Starting hostname init process")
    # result = c.run(docker_run_cmd())


if __name__ == "__main__":
    name_server_resolution = True
    ecs_resolution = False

    targets_subnet_path = path_settings.DATASET / "targets_subnet.json"
    vps_subnet_path = path_settings.DATASET / "vps_subnet.json"

    if not targets_subnet_path.exists():
        targets_subnet = load_target_subnets(clickhouse_settings.VPS_FILTERED)
        dump_json(targets_subnet, targets_subnet_path)
    if not vps_subnet_path.exists():
        vps_subnet = load_vp_subnets(clickhouse_settings.VPS_FILTERED)
        dump_json(vps_subnet, vps_subnet_path)

    targets_subnet = load_json(targets_subnet_path)
    vps_subnet = load_json(vps_subnet_path)

    gcp_vms = {
        # "iris-asia-east1": "35.206.250.197",
        # "iris-asia-northeast1": "35.213.102.165",
        # "iris-asia-southeast1": "35.213.136.86",
        # "iris-us-east4": "35.212.77.8",
        # "iris-southamerica-east1": "35.215.236.49",
        # "iris-asia-south1": "35.207.223.116",
        # "iris-europe-north1": "35.217.61.50",
        "iris-europe-west6": "35.216.205.173",
        # "iris-us-west4": "35.219.175.87",
        # "iris-me-central1": "34.1.33.16",
    }

    # generate mapping file from hostname file
    hostname_file = "hostname_per_org_per_ns.json"

    targets_table = clickhouse_settings.VPS_FILTERED
    vps_table = clickhouse_settings.VPS_FILTERED

    targets_ecs_table = "vps_mapping_ecs"
    vps_ecs_table = "vps_mapping_ecs"

    targets_mapping_path = (
        path_settings.DEFAULT / "../score_datasets/targets_mapping.pickle"
    )
    vps_mapping_path = path_settings.DEFAULT / "../score_datasets/vps_mapping.pickle"

    hostname_per_cdn_per_ns = load_json(path_settings.DATASET / hostname_file)

    main_org_thresholds = [0.2, 0.4, 0.6, 0.8]
    bgp_prefixes_thresholds = [2, 5, 10, 50, 1_00, 5_00]

    threshold_pairs = [(0.6, 10), (0.8, 10), (0.2, 10)]

    config_per_vm = {}
    for i, (vm, ip_addr) in enumerate(gcp_vms.items()):

        main_org_threshold, bgp_prefixes_threshold = threshold_pairs[i]

        selected_hostnames_per_cdn, selected_hostnames = load_hostnames(
            hostname_per_cdn_per_ns, main_org_threshold, bgp_prefixes_threshold
        )

        output_path = (
            path_settings.RESULTS_PATH
            / f"scores__10_hostname_per_cdn_per_ns_major_cdn_threshold_{main_org_threshold}_bgp_prefix_threshold_{bgp_prefixes_threshold}.pickle"
        )

        # upload vps and target mapping both to local vm folder and remote

        score_config = {
            "targets_table": targets_table,
            "vps_table": vps_table,
            "targets_subnet_path": str(targets_subnet_path),
            "vps_subnet_path": str(vps_subnet_path),
            "targets_mapping_path": str(targets_mapping_path),
            "vps_mapping_path": str(vps_mapping_path),
            "main_org_threshold": main_org_threshold,
            "bgp_prefixes_threshold": bgp_prefixes_threshold,
            "hostname_per_cdn_per_ns": hostname_per_cdn_per_ns,
            "hostname_per_cdn": selected_hostnames_per_cdn,
            "selected_hostnames": selected_hostnames,
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
            "output_path": str(output_path),
        }

        config_per_vm[vm] = {"ip_addr": ip_addr, "score_config": score_config}

    # logger.info(f"NB vms:: {len(gcp_vms)}")
    for vm, vm_config in config_per_vm.items():
        logger.info(
            f"{vm=}, {vm_config['ip_addr']=}, {len(vm_config['score_config']['selected_hostnames'])=}"
        )
        deploy_score(vm, vm_config)
        # monitor_memory_space(vm, vm_config)
        # rsync_files(vm, vm_config, delete_after=True)
        # check_docker_running(vm, vm_config)
