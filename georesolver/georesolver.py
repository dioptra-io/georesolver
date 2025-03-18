"""from a measurement config containing multiple agents, run georesolver"""

import subprocess

from pathlib import Path
from loguru import logger
from random import shuffle
from datetime import datetime

from georesolver.common.ssh_utils import ssh_run_cmd, ssh_upload_file, ssh_run_cmds
from georesolver.common.files_utils import (
    load_csv,
    dump_csv,
    load_json,
    dump_json,
    copy_to,
)
from georesolver.common.settings import (
    PathSettings,
    ClickhouseSettings,
    RIPEAtlasSettings,
)

path_settings = PathSettings()
ch_settings = ClickhouseSettings()
ripe_atlas_settings = RIPEAtlasSettings()


def check_input_dirs(target_file: Path, hostname_file: Path) -> None:
    """check that the input file exists"""
    if not target_file.exists():
        raise RuntimeError(f"{target_file} does not exists")

    if not hostname_file.exists():
        raise RuntimeError(f"{hostname_file} does not exists")


def check_agents(agents: list[dict]) -> set:
    """check if the config is valid, return all processes for table validation"""
    for agent_definition in agents:
        assert "host" in agent_definition
        assert "remote_dir" in agent_definition


def check_config(config_path: Path) -> None:
    """check the validity of the input config"""
    experiment_config = load_json(config_path, exit_on_failure=True)

    # mandatory parameters
    assert "target_file" in experiment_config
    assert "hostname_file" in experiment_config
    assert "experiment_uuid" in experiment_config
    assert "agents" in experiment_config

    # check if input files exist
    check_input_dirs(
        Path(experiment_config["target_file"]),
        Path(experiment_config["hostname_file"]),
    )

    # check agents definition
    check_agents(experiment_config["agents"])

    logger.info(f"Config:: {config_path} valid, can start experiment")

    return experiment_config


def create_experiment_path(experiment_uuid: str) -> Path:
    """create the experiment path"""
    experiment_path = path_settings.EXPERIMENT_PATH / f"{experiment_uuid}"
    if not experiment_path.is_dir():
        experiment_path.mkdir(parents=True, exist_ok=True)

    return experiment_path


def create_agent_path(agent_host: str, experiment_path: Path) -> Path:
    """create agent path"""
    agent_dir = experiment_path / agent_host
    if not agent_dir.is_dir():
        agent_dir.mkdir(parents=True, exist_ok=True)

    return agent_dir


def docker_run_agent_cmd(
    mount_path: str,
    agent_config_path: Path,
    container_name: str,
) -> str:
    """define the docker run command for starting a georesolver docker agent"""
    return f"""
        docker run -d \
        --name {container_name} \
        -v "{mount_path}:{mount_path}" \
        -v "{mount_path}/rib_table.dat:/app/georesolver/datasets/static_files/rib_table.dat"  \
        -e RIPE_ATLAS_SECRET_KEY={RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY} \
        -e CLICKHOUSE_HOST={ClickhouseSettings().CLICKHOUSE_HOST} \
        -e CLICKHOUSE_PORT={ClickhouseSettings().CLICKHOUSE_PORT} \
        -e CLICKHOUSE_DATABASE={ClickhouseSettings().CLICKHOUSE_DATABASE} \
        -e CLICKHOUSE_USERNAME={ClickhouseSettings().CLICKHOUSE_USERNAME} \
        -e CLICKHOUSE_PASSWORD={ClickhouseSettings().CLICKHOUSE_PASSWORD} \
        -e RIPE_ATLAS_SECRET_KEY={RIPEAtlasSettings().RIPE_ATLAS_SECRET_KEY} \
        --network host \
        --entrypoint poetry \
        ghcr.io/dioptra-io/georesolver:main run python georesolver/agent/main.py {mount_path}/{agent_config_path.name}
    """


def docker_pull_cmd() -> list[str]:
    return [
        f"echo {path_settings.GITHUB_TOKEN} | docker login ghcr.io -u {path_settings.DOCKER_USERNAME} --password-stdin",
        f"docker pull ghcr.io/dioptra-io/georesolver:main",
    ]


def print_docker_cmd(cmd: str) -> None:
    """print nicely the docker command that was executed"""
    cmd = cmd.split("  ")
    for row in cmd:
        if row:
            row = row.strip("\n").strip(" ")
            logger.debug(f"{row}")


def agent_start(
    user: str,
    host: str,
    local_dir: Path,
    remote_dir: Path,
    agent_config_path: Path,
    container_name: str,
    gateway: dict = None,
) -> None:
    """start georesolver on the remote server"""
    logger.info(f"Agent {host}:: start agent experiment")

    if host not in ["localhost", "127.0.0.1"]:
        cmd = docker_run_agent_cmd(
            remote_dir,
            agent_config_path,
            container_name,
        )

        _ = ssh_run_cmd(
            cmd,
            user,
            host,
            gateway["user"],
            gateway["host"],
        )

    else:
        cmd = docker_run_agent_cmd(
            local_dir,
            agent_config_path,
            container_name,
        )

        ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if ps.stderr:
            raise RuntimeError(f"Could not start local agent:: {ps.stderr}")

    logger.debug("Docker cmd::")
    print_docker_cmd(cmd)


def check_local_dir(
    user: str, host: str, target_file: Path, hostname_file: Path
) -> None:
    """check that the input file exists"""
    logger.info(f"Agent {host}:: checking local dirs")

    if not target_file.exists():
        raise RuntimeError(f"Agent={user}@{host}:: {target_file} does not exists")

    if not hostname_file.exists():
        raise RuntimeError(f"Agent={user}@{host}:: {hostname_file} does not exists")


def check_connection(
    user: str,
    host: str,
    gateway: dict,
) -> None:
    """various check for connection and file existence"""
    # test ssh connection to remote server
    connect_cmd = "echo agent connection ok"
    result = ssh_run_cmd(
        connect_cmd,
        user=user,
        host=host,
        gateway_user=gateway["user"],
        gateway_host=gateway["host"],
    )

    stdout = result.stdout.strip("\n")
    logger.info(f"Agent={user}@{host}:: {stdout}")


def create_remote_dir(user: str, host: str, gateway: dict, remote_dir: Path) -> None:
    """create remote working directory"""
    cmd = f"mkdir -p {remote_dir}"
    _ = ssh_run_cmd(
        cmd,
        user=user,
        host=host,
        gateway_user=gateway["user"],
        gateway_host=gateway["host"],
    )

    logger.info(f"Agent={user}@{host}:: remote dir={remote_dir} created")


def create_agent_config(
    host: str,
    agent_uuid: str,
    local_dir: Path,
    remote_dir: Path,
    target_file: Path,
    hostname_file: Path,
    processes: list,
    batch_size: int,
) -> Path:
    """create remote agent config, necessary for starting docker image"""
    if host not in ["localhost", "127.0.0.1"]:
        agent_config = {
            "agent_uuid": str(agent_uuid),
            "target_file": str(remote_dir / target_file.name),
            "hostname_file": str(remote_dir / hostname_file.name),
            "batch_size": batch_size,
            "processes": processes,
            "log_path": str(remote_dir / "logs"),
        }
    else:
        agent_config = {
            "agent_uuid": str(agent_uuid),
            "target_file": str(local_dir / target_file.name),
            "hostname_file": str(local_dir / hostname_file.name),
            "batch_size": batch_size,
            "processes": processes,
            "log_path": str(local_dir / "logs"),
        }

    # upload config file in local dir
    output_path = local_dir / "config.json"
    dump_json(agent_config, output_path)

    return output_path


def upload_target_files(
    user: str,
    host: str,
    gateway: dict,
    local_dir: Path,
    remote_dir: Path,
    files: list[Path],
) -> None:
    """upload target and hostname files to"""
    logger.info(f"Agent {host}:: target file updload")

    if host not in ["localhost", "127.0.0.1"]:

        for file in files:
            _ = ssh_upload_file(
                in_file=file,
                out_file=str(remote_dir / file.name),
                user=user,
                host=host,
                gateway_user=gateway["user"],
                gateway_host=gateway["host"],
            )
            logger.info(f"Agent={user}@{host}:: {file.name} upload done")

    else:
        # just copy the RIB table
        cmd = f"cp {path_settings.RIB_TABLE} {local_dir}"
        ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if ps.stderr:
            raise RuntimeError(f"Could not start local agent:: {ps.stderr}")


def pull_docker_image(user: str, host: str, gateway: str) -> None:
    """pull georesolver docker image"""
    # docker pull cmd common to local and remote
    logger.info(f"Agent {host}:: pull docker image")

    cmds = docker_pull_cmd()
    if host not in ["localhost", "127.0.0.1"]:

        _ = ssh_run_cmds(
            cmds,
            user=user,
            host=host,
            gateway_user=gateway["user"],
            gateway_host=gateway["host"],
        )
    else:
        for cmd in cmds:
            # TODO: general, better docker image management (terraform? kubernetes?)
            _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)


def run_remote_agent(
    agent_uuid: str,
    user: str,
    host: str,
    local_dir: Path,
    remote_dir: Path,
    target_file: Path,
    hostname_file: Path,
    processes: list[dict],
    batch_size: int = 1_000,
    max_ongoing_pings: int = 1_00,
    gateway: dict = None,
) -> None:
    """check connection, pull docker image, start main agent remotely"""
    # create null gateway if None was given
    gateway = gateway if gateway else {"user": None, "host": None}

    # Unused for now, meant for monitoring remote agent execution
    container_name = "georesolver__" + agent_uuid.replace("-", "_")

    # create remote agent config
    agent_config_path = create_agent_config(
        host=host,
        agent_uuid=agent_uuid,
        local_dir=local_dir,
        remote_dir=remote_dir,
        target_file=target_file,
        hostname_file=hostname_file,
        processes=processes,
        batch_size=batch_size,
    )

    # pull docker image
    check_local_dir(user, host, target_file, hostname_file)

    if host not in ["localhost", "127.0.0.1"]:
        check_connection(user, host, gateway)
        create_remote_dir(user, host, gateway, remote_dir)

    # upload target files
    # create remote agent config
    target_files = [
        agent_config_path,
        target_file,
        hostname_file,
        path_settings.RIB_TABLE.resolve(),
    ]
    upload_target_files(user, host, gateway, local_dir, remote_dir, target_files)

    # pull docker image
    pull_docker_image()

    # start docker container with params
    agent_start()

    # monitor container execution
    # TODO: too complex for experimental prototype, left for future dev
    # monitor()


def main(config_path: Path) -> None:
    """entry point, run georesolver"""
    # check config validity
    config = check_config(config_path)

    # add config start time
    config["start_time"] = str(datetime.now())

    # create experiement directory
    experiment_path = config["experiment_name"] + "__" + config["experiment_uuid"]
    experiment_path = create_experiment_path(experiment_path)

    # save general config in experiment path
    dump_json(config, experiment_path / config_path.name)

    # load targets from target file
    targets = load_csv(Path(config["target_file"]))
    shuffle(targets)

    # split measurement load equally among agents
    agent_target_load = len(targets) // len(config["agents"]) + 1
    agent_max_ping = ripe_atlas_settings.MAX_MEASUREMENT // len(config["agents"]) - 1
    for i, agent_definition in enumerate(config["agents"]):
        # create fresh directory for agent
        agent_dir = create_agent_path(agent_definition["host"], experiment_path)

        # upload local agent targets
        agent_targets = targets[i * agent_target_load : (i + 1) * agent_target_load]
        dump_csv(agent_targets, agent_dir / "targets.csv")
        copy_to(config["hostname_file"], agent_dir)

        # TOOD: run parrallel
        run_remote_agent(
            agent_uuid=agent_definition["agent_uuid"],
            user=agent_definition["user"],
            host=agent_definition["host"],
            local_dir=agent_dir,
            remote_dir=agent_definition["remote_dir"],
            target_file=agent_dir / "targets.csv",
            hostname_file=agent_dir / Path(config["hostname_file"]).name,
            processes=config["processes"],
            batch_size=config["batch_size"],
            max_ongoing_pings=agent_max_ping,
            gateway=(
                agent_definition["gateway"] if "gateway" in agent_definition else None
            ),
        )
