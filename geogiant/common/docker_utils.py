"""
define a set of function to manipulate, run and get information about running docker container
note: these functions does not use docker python module as most docker images are running remotely
"""

from pathlib import Path
from loguru import logger

from geogiant.common.ssh_utils import ssh_run_cmd
from geogiant.common.settings import PathSettings, RIPEAtlasSettings, ClickhouseSettings

path_settings = PathSettings()
ripe_atlas_settings = RIPEAtlasSettings()
clickhouse_settings = ClickhouseSettings()


def docker_run_agent_cmd(remote_dir: str, agent_config_path: Path) -> str:
    """define the docker run command for starting a georesolver docker agent"""
    return f"""
        docker run -d \
        -v "{remote_dir}:/app/geogiant/experiments/" \
        -v "{remote_dir}/rib_table.dat:/app/geogiant/datasets/static_files/rib_table.dat"  \
        -e RIPE_ATLAS_SECRET_KEY={ripe_atlas_settings.RIPE_ATLAS_SECRET_KEY} \
        -e CLICKHOUSE_HOST={clickhouse_settings.CLICKHOUSE_HOST} \
        -e CLICKHOUSE_PORT={clickhouse_settings.CLICKHOUSE_PORT} \
        -e CLICKHOUSE_DATABASE={clickhouse_settings.CLICKHOUSE_DATABASE} \
        -e CLICKHOUSE_USERNAME={clickhouse_settings.CLICKHOUSE_USERNAME} \
        -e CLICKHOUSE_PASSWORD={clickhouse_settings.CLICKHOUSE_PASSWORD} \
        -e RIPE_ATLAS_SECRET_KEY={ripe_atlas_settings.RIPE_ATLAS_SECRET_KEY} \
        --network host \
        --entrypoint poetry \
        ghcr.io/dioptra-io/geogiant:main run python geogiant/main.py /app/geogiant/experiments/{agent_config_path.name}
    """


def docker_pull_cmd() -> list[str]:
    return [
        f"echo {path_settings.GITHUB_TOKEN} | docker login ghcr.io -u {path_settings.DOCKER_USERNAME} --password-stdin",
        f"docker pull ghcr.io/dioptra-io/geogiant:main",
    ]


def get_running_images(agent_config: dict) -> str:
    """check if docker image is running or not"""
    # TODO: parse string output
    result = ssh_run_cmd(
        cmd="docker ps",
        user=agent_config["user"],
        host=agent_config["host"],
        gateway_host=(
            agent_config["gateway_host"] if "gateway_host" in agent_config else None
        ),
        gateway_user=(
            agent_config["gateway_user"] if "gateway_user" in agent_config else None
        ),
        exit_on_failure=True,
    )

    logger.debug(f"docker images running running:: {result.stdout}")

    return result


def free_memory(agent_config: dict) -> None:
    """VMs storage is small, after rsync, remove remote csv files"""
    logger.info(f"Freeing memory on vm:: {vm}")

    c = Connection(f"{path_settings.SSH_USER}@{vm_config['ip_addr']}")
    result = c.run("rm -rf results/name_server_resolution_*")


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
    selected_hostnames_file = vm_result_path / "all_ecs_selected_hostnames.csv"
    dump_csv(vm_config["hostnames"], vm_result_path / "all_ecs_selected_hostnames.csv")

    vm_config_file_path = vm_result_path / "vm_config.json"
    dump_json(vm_config, output_file=vm_config_file_path)

    # upload hostname file
    result = c.put(
        local=f"{selected_hostnames_file}",
        remote="datasets",
    )

    # upload hostname file
    result = c.put(
        local=f"{vm_config_file_path}",
        remote="datasets",
    )

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
        for vm, ip_addr in gcp_vms.items():
            config_per_vm[vm] = {"ip_addr": ip_addr, "hostnames": ecs_hostnames}

    if ecs_resolution:
        resolved_hostnames = get_resolved_hostnames("filtered_hostnames_ecs_mapping")
        ecs_hostnames = load_csv(
            path_settings.DATASET / "all_ecs_selected_hostnames.csv"
        )
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
        monitor_memory_space(vm, vm_config)
        rsync_files(vm, vm_config, delete_after=True)
        check_docker_running(vm, vm_config)

    insert_ecs_mapping_results(gcp_vms, output_table="vps_mapping_ecs")
    insert_local_results(
        local_table="filtered_hostnames_ecs_mapping",
        remote_table="vps_mapping_ecs",
        output_table="vps_mapping_ecs",
    )

    insert_name_server_results(gcp_vms, output_table="name_servers")
