import subprocess

from random import shuffle
from loguru import logger
from fabric import Connection
from ipaddress import IPv4Address, AddressValueError
from pych_client import ClickHouseClient

from geogiant.clickhouse import (
    GetHostnames,
    CreateDNSMappingTable,
    InsertFromCSV,
)
from geogiant.common.files_utils import (
    load_csv,
    dump_csv,
    dump_json,
    load_json_iter,
    load_json,
)
from geogiant.common.ip_addresses_utils import get_prefix_from_ip
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

END_TO_END_HOSTNAME_PATH = path_settings.END_TO_END_DATASET / "end_to_end_hostnames.csv"
END_TO_END_SUBNETS_PATH = path_settings.END_TO_END_DATASET / "end_to_end_subnets.json"
DOCKER_IMAGE_NAME = "geogiant-agent"


def docker_run_cmd() -> str:
    return f"""
        docker run -d \
        -v "$(pwd)/results:/app/geogiant/results" \
        -v "$(pwd)/datasets/selected_hostnames.csv:/app/geogiant/datasets/selected_hostnames.csv" \
        -v "$(pwd)/datasets/end_to_end_subnets.json:/app/geogiant/datasets/end_to_end_subnets.json" \
        --network host \
        --name {DOCKER_IMAGE_NAME} \
        ghcr.io/dioptra-io/geogiant:main
    """


def check_docker_running(vm: str, vm_config: dict) -> None:
    """check if docker image is running or not"""
    logger.info(f"Check if docker image running on:: {vm}")

    try:
        c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")
    except:
        logger.error(f"Could not connect to VM:: {vm}:{vm_config['ip_addr']}")

    result = c.run(f"docker ps -f name={DOCKER_IMAGE_NAME} --format json", hide=True)

    if result.stdout:
        logger.info(f"{DOCKER_IMAGE_NAME} running on {vm}")
    else:
        logger.info(f"{DOCKER_IMAGE_NAME} NOT running on {vm}")


def insert_ecs_mapping_results(gcp_vms: dict, output_table: str) -> None:
    """insert csv results file into clickhouse"""
    # insert files
    for vm in gcp_vms:
        vm_results_path = path_settings.RESULTS_PATH / f"{vm}"

        result_files = [
            file
            for file in vm_results_path.iterdir()
            if "end_to_end_ecs_resolution" in file.name
        ]

        logger.info(f"Inserting mapping results for VM:: {vm}")

        for file in result_files:

            logger.info(f"Inserting file:: {file}")

            with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
                CreateDNSMappingTable().execute(client=client, table_name=output_table)

                InsertFromCSV().execute_from_in_file(
                    table_name=output_table,
                    in_file=file,
                )


def free_memory(vm: str, vm_config: dict) -> None:
    """VMs storage is small, after rsync, remove remote csv files"""
    logger.info(f"Freeing memory on vm:: {vm}")

    c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")
    result = c.run("rm -rf results/end_to_end_ecs_resolution_*")


def rsync_files(vm: str, vm_config: dict, delete_after: bool = False) -> None:
    """rsync result file"""
    vm_result_path = path_settings.RESULTS_PATH / f"{vm}"

    # rsync remote result dir
    result = subprocess.run(
        f"rsync -av -e 'ssh -o StrictHostKeyChecking=no' {path_settings.SSH_USER}@{vm_config['ip_addr']}:results/ {vm_result_path}/",
        shell=True,
        capture_output=True,
        text=True,
    )

    if delete_after:
        free_memory(vm, vm_config)

    logger.info(f"Rsynced with remote dir:: {vm}")
    logger.info(result)


def monitor_memory_space(vm: str, vm_config: dict) -> None:
    """check memory space available"""
    c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")
    result = c.run("df -h /home/hugo/")
    logger.info(f"Memory available for VM {vm}:: {result}")


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

    # dump ecs hostname file for VM
    selected_hostnames_file = vm_result_path / "selected_hostnames.csv"
    dump_csv(vm_config["hostnames"], vm_result_path / "selected_hostnames.csv")

    vm_subnet_file_path = vm_result_path / "end_to_end_subnets.json"
    dump_json(vm_config["subnets"], vm_result_path / "end_to_end_subnets.json")

    # upload hostname file
    result = c.put(
        local=f"{selected_hostnames_file}",
        remote="datasets",
    )

    # upload subnets file
    result = c.put(
        local=f"{vm_subnet_file_path}",
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
    name_server_resolution = True
    ecs_resolution = False

    # "iris-me-central1": "34.1.33.16",
    # "iris-asia-southeast1": "35.213.136.86",

    gcp_vms = {
        "iris-europe-north1": "35.217.17.6",
        "iris-us-east4": "35.212.12.175",
        "iris-europe-west6": "35.216.186.30",
        "iris-us-west4": "35.219.147.41",
        "iris-southamerica-east1": "35.215.234.244",
        "iris-asia-south1": "35.207.233.237",
        "iris-asia-east1": "35.213.132.83",
        "iris-asia-northeast1": "35.213.95.10",
    }

    # load hostnames and subnets
    selected_hostnames = load_csv(END_TO_END_HOSTNAME_PATH)
    routers_subnet = load_json(END_TO_END_SUBNETS_PATH)

    logger.info(f"Total number of selected hostnames:: {len(selected_hostnames)}")
    logger.info(f"Total number of selected subnets:: {len(routers_subnet)}")

    # load routers subnets per VMs
    subnets_per_vm = []
    batch_size = len(routers_subnet) // len(gcp_vms) + 1
    for i in range(0, len(routers_subnet), batch_size):
        subnets_per_vm.append(routers_subnet[i : i + batch_size])

    config_per_vm = {}
    for i, (vm, ip_addr) in enumerate(gcp_vms.items()):
        config_per_vm[vm] = {
            "ip_addr": ip_addr,
            "hostnames": selected_hostnames,
            "subnets": subnets_per_vm[i],
        }

    logger.info(f"NB vms:: {len(gcp_vms)}")
    for vm, vm_config in config_per_vm.items():
        logger.info(
            f"{vm=}, {vm_config['ip_addr']=}, {len(vm_config['hostnames'])=}, {len(vm_config['subnets'])}"
        )
        # deploy_hostname_resolution(vm, vm_config)
        check_docker_running(vm, vm_config)
        # monitor_memory_space(vm, vm_config)
        # rsync_files(vm, vm_config, delete_after=False)
        # time.sleep(5)

    # insert_ecs_mapping_results(gcp_vms, output_table="end_to_end_mapping_ecs")
