import subprocess

from loguru import logger
from fabric import Connection
from pych_client import ClickHouseClient

from geogiant.clickhouse import GetHostnames
from geogiant.common.files_utils import load_csv, dump_csv
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def docker_run_cmd() -> str:
    return """
        docker run -d \
        -v "$(pwd)/results:/app/geogiant/results" \
        -v "$(pwd)/datasets/ecs_selected_hostnames.csv:/app/geogiant/datasets/ecs_selected_hostnames.csv" \
        ghcr.io/dioptra-io/geogiant:main
    """


# load resolved hostnames
def get_resolved_hostnames() -> list:
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        hostnames = GetHostnames().execute(client, clickhouse_settings.DNS_MAPPING_VPS)

    hostnames = [hostname["hostname"] for hostname in hostnames]

    return hostnames


def deploy_hostname_resolution(vm: str, vm_config: dict) -> None:
    """run docker image on gcp VMs"""
    logger.info(f"Running ECS resolution on:: {vm}")

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

    # # rsync remote result dir
    # result = subprocess.run(
    #     f"rsync -av -e \"ssh -o StrictHostKeyChecking=no\" {path_settings.SSH_USER}@{vm_config['ip_addr']}:/home/{path_settings.SSH_USER}/results/ {vm_result_path}/",
    #     shell=True,
    #     capture_output=True,
    #     text=True,
    # )

    logger.info(f"Rsynced with remote dir")

    # dump ecs hostname file for VM
    selected_hostnames_file = vm_result_path / "ecs_selected_hostnames.csv"
    dump_csv(vm_config["hostnames"], vm_result_path / "ecs_selected_hostnames.csv")

    # upload hostname file
    result = c.put(
        local=f"{selected_hostnames_file}",
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
    logger.info("Starting hostname init process")
    result = c.run(docker_run_cmd())


if __name__ == "__main__":
    resolved_hostnames = get_resolved_hostnames()
    ecs_hostnames = load_csv(path_settings.DATASET / "ecs_selected_hostnames.csv")
    remaining_hostnames = list(
        set(resolved_hostnames).symmetric_difference(set(ecs_hostnames))
    )

    logger.info(f"Total number of ECS hostnames:: {len(ecs_hostnames)}")
    logger.info(f"Resolved ECS hostnames:: {len(resolved_hostnames)}")
    logger.info(f"Remaining hostnames:: {len(remaining_hostnames)}")

    gcp_vms = {
        "iris-asia-east1": "35.206.250.197",
        "iris-asia-northeast1": "35.213.102.165",
        "iris-asia-southeast1": "35.213.136.86",
        "iris-us-east4": "35.212.77.8",
        "iris-southamerica-east1": "35.215.236.49",
        "iris-asia-south1": "35.207.223.116",
        "iris-europe-north1": "35.217.61.50",
        "iris-europe-west6": "35.216.205.173",
        "iris-us-west4": "35.219.175.87",
        "iris-me-central1": "34.1.33.16",
    }

    hostname_per_vm = []
    batch_size = len(remaining_hostnames) // len(gcp_vms) + 1
    for i in range(0, len(remaining_hostnames), batch_size):
        hostname_per_vm.append(remaining_hostnames[i : i + batch_size])

    config_per_vm = {}
    for i, (vm, ip_addr) in enumerate(gcp_vms.items()):
        config_per_vm[vm] = {"ip_addr": ip_addr, "hostnames": hostname_per_vm[i]}

    logger.info(f"NB vms:: {len(gcp_vms)}")
    for vm, vm_config in config_per_vm.items():
        logger.info(f"{vm=}, {vm_config['ip_addr']=}, {len(vm_config['hostnames'])=}")
        deploy_hostname_resolution(vm, vm_config)
