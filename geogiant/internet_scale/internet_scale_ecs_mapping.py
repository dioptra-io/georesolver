import subprocess
import asyncio

from random import shuffle
from loguru import logger
from fabric import Connection
from pych_client import ClickHouseClient

from geogiant.clickhouse import (
    CreateDNSMappingTable,
    InsertFromCSV,
)
from geogiant.common.files_utils import (
    load_csv,
    dump_csv,
    dump_json,
    load_json,
    create_tmp_json_file,
)
from geogiant.common.queries import get_subnets
from geogiant.ecs_mapping_init import resolve_vps_subnet
from geogiant.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()

INTERNET_SCALE_HOSTNAME_PATH = (
    path_settings.INTERNET_SCALE_DATASET / "internet_scale_hostnames.csv"
)
INTERNET_SCALE_SUBNETS_PATH = (
    path_settings.INTERNET_SCALE_DATASET / "internet_scale_subnets.json"
)
ECS_TABLE = "internet_scale_mapping_ecs"
DOCKER_IMAGE_NAME = "geogiant-agent"


def docker_run_cmd() -> str:
    return f"""
        docker run -d \
        -v "$(pwd)/results:/app/geogiant/results" \
        -v "$(pwd)/datasets/selected_hostnames.csv:/app/geogiant/datasets/selected_hostnames.csv" \
        -v "$(pwd)/datasets/internet_scale_subnets.json:/app/geogiant/datasets/internet_scale_subnets.json" \
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
            if "internet_scale_ecs_resolution" in file.name
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
    c.run("rm -rf results/internet_scale_ecs_resolution_*")


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

    vm_subnet_file_path = vm_result_path / "internet_scale_subnets.json"
    dump_json(vm_config["subnets"], vm_result_path / "internet_scale_subnets.json")

    # upload hostname file
    c.put(
        local=f"{selected_hostnames_file}",
        remote="datasets",
    )

    # upload subnets file
    c.put(
        local=f"{vm_subnet_file_path}",
        remote="datasets",
    )

    # docker login
    c.run(f"export PAT={path_settings.GITHUB_TOKEN}")
    c.run(
        f"docker login ghcr.io -u {path_settings.DOCKER_USERNAME} -p {path_settings.GITHUB_TOKEN}"
    )

    # pull docker image
    c.run("docker pull ghcr.io/dioptra-io/geogiant:main")

    # run docker image
    logger.info("Starting hostname init process")
    c.run(docker_run_cmd())


def distributed_ecs_mapping() -> None:

    gcp_vms = {
        "iris-europe-north1": "35.217.17.6",
        "iris-us-east4": "35.212.12.175",
        "iris-europe-west6": "35.216.186.30",
        "iris-us-west4": "35.219.147.41",
        "iris-southamerica-east1": "35.215.234.244",
        "iris-asia-south1": "35.207.233.237",
        "iris-asia-east1": "35.213.132.83",
        "iris-asia-northeast1": "35.213.95.10",
        "iris-me-central1": "34.1.33.16",
        "iris-asia-southeast1": "35.213.136.86",
    }

    # load hostnames and subnets
    selected_hostnames = load_csv(INTERNET_SCALE_HOSTNAME_PATH)
    routers_subnet = load_json(INTERNET_SCALE_SUBNETS_PATH)

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

    # insert_ecs_mapping_results(gcp_vms, output_table="internet_scale_mapping_ecs")


def load_hostnames() -> list[str]:
    selected_hostnames = set()

    pair_config = [(20, 3)]
    for bgp_threshold, nb_hostname_per_ns_org in pair_config:
        selected_hostnames_per_cdn_per_ns = load_json(
            path_settings.DATASET
            / f"hostname_geo_score_selection_{bgp_threshold}_BGP_{nb_hostname_per_ns_org}_hostnames_per_org_ns.json"
        )

        hostnames_per_config = set()
        for ns in selected_hostnames_per_cdn_per_ns:
            for _, hostnames in selected_hostnames_per_cdn_per_ns[ns].items():
                selected_hostnames.update(hostnames)
                hostnames_per_config.update(hostnames)

        logger.debug(
            f"{bgp_threshold=}, {nb_hostname_per_ns_org=}, {len(hostnames_per_config)} hostnames"
        )

        logger.info(f"Nb selected hostnames:: {len(selected_hostnames)}")

    logger.info(f"Total Nb selected hostnames:: {len(selected_hostnames)}")

    return selected_hostnames


def load_subnets(subnet_batch_size: int = 10_000) -> list[str]:
    if not INTERNET_SCALE_SUBNETS_PATH.exists():
        internet_scale_dataset = load_json(path_settings.USER_HITLIST_FILE)

        internet_scale_subnets = [subnet for subnet in internet_scale_dataset]
        shuffle(internet_scale_subnets)
        dump_json(internet_scale_subnets, INTERNET_SCALE_SUBNETS_PATH)

    internet_scale_subnets = load_json(INTERNET_SCALE_SUBNETS_PATH)
    cached_subnets = get_subnets(ECS_TABLE)

    logger.info(f"Total number of hostnames:: {len(internet_scale_subnets)}")

    remaining_subnets = list(
        set(internet_scale_subnets).difference(set(cached_subnets))
    )
    shuffle(remaining_subnets)

    logger.info(f"Remaining number of hostnames:: {len(remaining_subnets)}")

    return remaining_subnets[:subnet_batch_size]


def load_hostnames() -> list[str]:
    if not INTERNET_SCALE_HOSTNAME_PATH.exists():
        selected_hostnames = load_hostnames()
        dump_csv(selected_hostnames, INTERNET_SCALE_HOSTNAME_PATH)

    return load_csv(INTERNET_SCALE_HOSTNAME_PATH)


async def local_ecs_mapping(subnet_batch_size: int = 1_000_000) -> None:
    internet_scale_subnets = load_subnets(subnet_batch_size)
    internets_scale_hostnames = load_hostnames()

    batch_size = 10_000
    for i in range(0, len(internet_scale_subnets), batch_size):
        logger.info(f"Batch:: {i+1}/{len(internet_scale_subnets) // batch_size}")
        batch_subnets = internet_scale_subnets[i : i + batch_size]

        subnet_tmp_file_path = create_tmp_json_file(batch_subnets)

        await resolve_vps_subnet(
            selected_hostnames=internets_scale_hostnames,
            input_file=subnet_tmp_file_path,
            output_table=ECS_TABLE,
            chunk_size=500,
        )

        subnet_tmp_file_path.unlink()


async def main() -> None:
    distributed = False
    local = True

    if distributed:
        distributed_ecs_mapping()
    if local:
        await local_ecs_mapping()


if __name__ == "__main__":
    asyncio.run(main())
