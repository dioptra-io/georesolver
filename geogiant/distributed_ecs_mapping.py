import subprocess

from tqdm import tqdm
from loguru import logger
from fabric import Connection
from pych_client import ClickHouseClient

from geogiant.clickhouse import (
    GetHostnames,
    GetAllDNSMapping,
    CreateDNSMappingTable,
    InsertFromCSV,
)
from geogiant.common.files_utils import load_csv, dump_csv, create_tmp_csv_file
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


def check_docker_running(vm: str, vm_config: dict) -> None:
    """check if docker image is running or not"""
    logger.info(f"Check if docker image running on:: {vm}")

    c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")
    result = c.run("docker ps")


def insert_remote_results(gcp_vms: dict, output_table: str) -> None:
    """insert csv results file into clickhouse"""
    # insert files
    for vm in gcp_vms:
        vm_results_path = path_settings.RESULTS_PATH / f"{vm}"

        for file in vm_results_path.iterdir():
            if "vps_mapping_ecs_resolution_" in file.name:

                logger.info(f"Inserting file:: {file}")

                with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
                    CreateDNSMappingTable().execute(
                        client=client, table_name=output_table
                    )

                    InsertFromCSV().execute_from_in_file(
                        table_name=output_table,
                        in_file=file,
                    )


def get_all_dns_mapping(input_table: str, hostname_filter: list[str] = []):
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        rows = GetAllDNSMapping().execute_iter(
            client, input_table, hostname_filter=hostname_filter
        )

        for row in rows:
            yield row


def get_resolved_hostnames(input_table: str) -> list:
    with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
        hostnames = GetHostnames().execute(client, input_table)

    hostnames = [hostname["hostname"] for hostname in hostnames]

    return hostnames


def insert_local_results(
    local_table: str, remote_table: str, output_table: str
) -> None:
    """get all results that were outputed in local table and merge with remote results"""
    logger.info(f"Inserting rows from locally resolved results:: {local_table}")

    batch = 1000
    local_resolved_hostnames = get_resolved_hostnames(local_table)
    remote_resolved_hostnames = get_resolved_hostnames(remote_table)

    logger.info(f"{len(local_resolved_hostnames)}")
    logger.info(f"{len(remote_resolved_hostnames)}")

    hostnames_to_insert = list(
        set(local_resolved_hostnames).difference(set(remote_resolved_hostnames))
    )

    logger.debug(
        f"Nb hostname to insert from local to remote table:: {len(hostnames_to_insert)}"
    )
    for i in tqdm(range(0, len(hostnames_to_insert), batch)):
        hostname_filter = hostnames_to_insert[i : i + batch]
        rows = get_all_dns_mapping(local_table, hostname_filter=hostname_filter)

        locally_mapping_filtered = []
        for row in rows:
            row = ",".join([str(val) for val in row.values()])
            locally_mapping_filtered.append(row)

        # # generate tmp file
        tmp_file_path = create_tmp_csv_file(locally_mapping_filtered)

        with ClickHouseClient(**clickhouse_settings.clickhouse) as client:
            CreateDNSMappingTable().execute(client, output_table)

            InsertFromCSV().execute_from_in_file(output_table, tmp_file_path)

        tmp_file_path.unlink()


def free_memory(vm: str, vm_config: dict) -> None:
    """VMs storage is small, after rsync, remove remote csv files"""
    logger.info(f"Freeing memory on vm:: {vm}")

    c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")
    result = c.run("rm -rf results/vps_mapping_ecs_resolution_*")


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
    name_server_resolution = True
    ecs_resolution = False

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
    if name_server_resolution:
        ecs_hostnames = load_csv(path_settings.DATASET / "ecs_selected_hostnames.csv")

        logger.info(f"Total number of ECS hostnames:: {len(ecs_hostnames)}")

        config_per_vm = {}
        for i, (vm, ip_addr) in enumerate(gcp_vms.items()):
            config_per_vm[vm] = {"ip_addr": ip_addr, "hostnames": ecs_hostnames[i]}

    if ecs_resolution:
        resolved_hostnames = get_resolved_hostnames("filtered_hostnames_ecs_mapping")
        ecs_hostnames = load_csv(path_settings.DATASET / "ecs_selected_hostnames.csv")
        remaining_hostnames = list(
            set(resolved_hostnames).symmetric_difference(set(ecs_hostnames))
        )

        logger.info(f"Total number of ECS hostnames:: {len(ecs_hostnames)}")
        logger.info(f"Resolved ECS hostnames:: {len(resolved_hostnames)}")
        logger.info(f"Remaining hostnames:: {len(remaining_hostnames)}")

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
        # monitor_memory_space(vm, vm_config)
        # rsync_files(vm, vm_config, delete_after=True)
        # check_docker_running(vm, vm_config)

    # insert_remote_results(gcp_vms, output_table="vps_mapping_ecs")
    # insert_local_results(
    #     local_table="filtered_hostnames_ecs_mapping",
    #     remote_table="vps_mapping_ecs",
    #     output_table="vps_mapping_ecs",
    # )
