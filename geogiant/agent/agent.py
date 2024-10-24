"""define the agent class and functions"""

import subprocess

from enum import Enum
from time import sleep
from pathlib import Path
from loguru import logger
from pprint import pformat

from geogiant.common.files_utils import dump_json
from geogiant.common.ssh_utils import ssh_run_cmd, ssh_run_cmds, ssh_upload_file
from geogiant.common.settings import PathSettings, ClickhouseSettings, RIPEAtlasSettings

path_settings = PathSettings()
clickhouse_settings = RIPEAtlasSettings()
ripe_atlas_settings = RIPEAtlasSettings()


class ProcessNames(Enum):
    ECS_PROC = "ecs_process"
    SCORE_PROC = "score_process"
    PING_PROC = "ping_process"
    INSERT_PROC = "insert_process"


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
        -v "{mount_path}/rib_table.dat:/app/geogiant/datasets/static_files/rib_table.dat"  \
        -e RIPE_ATLAS_SECRET_KEY={ripe_atlas_settings.RIPE_ATLAS_SECRET_KEY} \
        -e CLICKHOUSE_HOST={clickhouse_settings.CLICKHOUSE_HOST} \
        -e CLICKHOUSE_PORT={clickhouse_settings.CLICKHOUSE_PORT} \
        -e CLICKHOUSE_DATABASE={clickhouse_settings.CLICKHOUSE_DATABASE} \
        -e CLICKHOUSE_USERNAME={clickhouse_settings.CLICKHOUSE_USERNAME} \
        -e CLICKHOUSE_PASSWORD={clickhouse_settings.CLICKHOUSE_PASSWORD} \
        -e RIPE_ATLAS_SECRET_KEY={ripe_atlas_settings.RIPE_ATLAS_SECRET_KEY} \
        --network host \
        --entrypoint poetry \
        ghcr.io/dioptra-io/geogiant:main run python geogiant/main.py {mount_path}/{agent_config_path.name}
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


class Agent:
    def __init__(
        self,
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

        self.agent_uuid = agent_uuid
        self.user = user
        self.host = host
        self.local_dir = Path(local_dir).resolve()
        self.remote_dir = Path(remote_dir).resolve() / agent_uuid
        self.target_file = Path(target_file).resolve()
        self.hostname_file = Path(hostname_file).resolve()
        self.processes = processes
        self.batch_size = batch_size
        self.max_ongoing_pings = max_ongoing_pings

        # create null gateway if None was given
        self.gateway = gateway if gateway else {"user": None, "host": None}

        # create remote agent config
        self.agent_config_path = self.create_agent_config()

        # create docker container name
        self.container_name = "georesolver__" + self.agent_uuid.replace("-", "_")

    def create_agent_config(self) -> Path:
        """create remote agent config, necessary for starting docker image"""
        if self.host not in ["localhost", "127.0.0.1"]:
            agent_config = {
                "agent_uuid": str(self.agent_uuid),
                "target_file": str(self.remote_dir / self.target_file.name),
                "hostname_file": str(self.remote_dir / self.hostname_file.name),
                "batch_size": self.batch_size,
                "processes": self.processes,
                "log_path": str(self.remote_dir / "logs"),
            }
        else:
            agent_config = {
                "agent_uuid": str(self.agent_uuid),
                "target_file": str(self.local_dir / self.target_file.name),
                "hostname_file": str(self.local_dir / self.hostname_file.name),
                "batch_size": self.batch_size,
                "processes": self.processes,
                "log_path": str(self.local_dir / "logs"),
            }

        # upload config file in local dir
        output_path = self.local_dir / "config.json"
        dump_json(agent_config, output_path)

        return output_path

    def check_connection(self) -> None:
        """various check for connection and file existence"""
        # test ssh connection to remote server
        connect_cmd = "echo agent connection ok"
        result = ssh_run_cmd(
            connect_cmd,
            user=self.user,
            host=self.host,
            gateway_user=self.gateway["user"],
            gateway_host=self.gateway["host"],
        )

        stdout = result.stdout.strip("\n")
        logger.info(f"Agent={self.user}@{self.host}:: {stdout}")

    def check_local_dir(self) -> None:
        """check that the input file exists"""
        if not self.target_file.exists():
            raise RuntimeError(
                f"Agent={self.user}@{self.host}:: {self.target_file} does not exists"
            )

        if not self.hostname_file.exists():
            raise RuntimeError(
                f"Agent={self.user}@{self.host}:: {self.hostname_file} does not exists"
            )

    def create_remote_dir(self) -> None:
        """create remote working directory"""
        cmd = f"mkdir -p {self.remote_dir}"
        _ = ssh_run_cmd(
            cmd,
            user=self.user,
            host=self.host,
            gateway_user=self.gateway["user"],
            gateway_host=self.gateway["host"],
        )

        logger.info(
            f"Agent={self.user}@{self.host}:: remote dir={self.remote_dir} created"
        )

    def upload_target_files(self, files: list[Path]) -> None:
        """upload target and hostname files to"""
        if self.host not in ["localhost", "127.0.0.1"]:

            for file in files:
                _ = ssh_upload_file(
                    in_file=file,
                    out_file=str(self.remote_dir / file.name),
                    user=self.user,
                    host=self.host,
                    gateway_user=self.gateway["user"],
                    gateway_host=self.gateway["host"],
                )
                logger.info(f"Agent={self.user}@{self.host}:: {file.name} upload done")

        else:
            # just copy the RIB table
            cmd = f"cp {path_settings.RIB_TABLE} {self.local_dir}"
            ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if ps.stderr:
                raise RuntimeError(f"Could not start local agent:: {ps.stderr}")

    def pull_docker_image(self) -> None:
        """pull georesolver docker image"""
        # docker pull cmd common to local and remote
        cmds = docker_pull_cmd()
        if self.host not in ["localhost", "127.0.0.1"]:

            _ = ssh_run_cmds(
                cmds,
                user=self.user,
                host=self.host,
                gateway_user=self.gateway["user"],
                gateway_host=self.gateway["host"],
            )
        else:
            for cmd in cmds:
                # TODO: general, better docker image management (terraform? kubernetes?)
                _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    def agent_start(self) -> None:
        """start georesolver on the remote server"""
        if self.host not in ["localhost", "127.0.0.1"]:
            cmd = docker_run_agent_cmd(
                self.remote_dir,
                self.agent_config_path,
                self.container_name,
            )

            logger.info(f"Starting remote agent :: {self.user}@{self.host}")

            _ = ssh_run_cmd(
                cmd,
                self.user,
                self.host,
                self.gateway["user"],
                self.gateway["host"],
            )

        else:
            cmd = docker_run_agent_cmd(
                self.local_dir,
                self.agent_config_path,
                self.container_name,
            )

            logger.info(f"Starting local agent")

            ps = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if ps.stderr:
                raise RuntimeError(f"Could not start local agent:: {ps.stderr}")

        logger.debug("Docker cmd::\n{}", pformat(cmd))

    def is_container_running(self):
        cmd = f"docker ps --filter name={self.container_name} --format {{{{.Names}}}}"
        try:
            # Run the 'docker ps' command to list running containers and check if the given container name is present
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
            )

            # Check if the output contains the container name
            running_containers = result.stdout.strip().split("\n")
            return self.container_name in running_containers

        except subprocess.CalledProcessError as e:
            # Handle errors from the 'docker' command
            logger.error(f"Error checking container status: {e}")
            return False

    def run(self) -> None:
        """start docker run on remote server (TODO: locally as well)"""
        self.check_local_dir()

        if self.host not in ["localhost", "127.0.0.1"]:
            # check if remote connection is ok
            self.check_connection()

            # create working dir remotely
            self.create_remote_dir()

        # upload target files
        # create remote agent config
        target_files = [
            self.agent_config_path,
            self.target_file,
            self.hostname_file,
            path_settings.RIB_TABLE,
        ]
        self.upload_target_files(target_files)

        # pull docker image
        self.pull_docker_image()

        # start docker container with params
        self.agent_start()

        # check docker running
        container_running = True
        while container_running:
            container_running = self.is_container_running()
            sleep(1)

        logger.info("Container stopped, measurement done")
