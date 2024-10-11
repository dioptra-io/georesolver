"""define the agent class and functions"""

from pathlib import Path
from loguru import logger
from enum import Enum
from pprint import pprint

from geogiant.common.files_utils import load_json
from geogiant.common.docker_utils import docker_pull_cmd, docker_run_agent_cmd
from geogiant.common.ssh_utils import ssh_run_cmd, ssh_run_cmds, ssh_upload_file
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


class ProcessNames(Enum):
    ECS_PROC = "ecs_process"
    SCORE_PROC = "score_process"
    PING_PROC = "ping_process"
    INSERT_PROC = "insert_process"


class Agent:
    def __init__(self, agent_config_path: Path) -> None:
        self.agent_config_path = agent_config_path
        self.agent_config = load_json(agent_config_path, exit_on_failure=True)

        # remote server ssh parameters
        try:
            self.user = self.agent_config["user"]
            self.host = self.agent_config["host"]
            self.local_dir = Path(self.agent_config["local_dir"])
            self.remote_dir = Path(self.agent_config["remote_dir"])
            self.target_file = Path(self.agent_config["target_file"])
            self.hostname_file = Path(self.agent_config["hostname_file"])
        except KeyError as e:
            raise RuntimeError(f"Agent parameter {e} is missing")

        # in the case of ssh proxy jump, not nested
        if gateway := self.agent_config.get("gateway"):
            self.gateway = gateway
        else:
            self.gateway = {"user": None, "host": None}

        # mandatory parameters check
        assert "processes" in self.agent_config
        assert "max_ongoing_ping" in self.agent_config

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

    def upload_target_files(self) -> None:
        """upload target and hostname files to"""
        target_files = [
            self.agent_config_path,
            self.target_file,
            self.hostname_file,
            path_settings.RIB_TABLE,
        ]

        for file in target_files:
            _ = ssh_upload_file(
                in_file=file,
                out_file=str(self.remote_dir / file.name),
                user=self.user,
                host=self.host,
                gateway_user=self.gateway["user"],
                gateway_host=self.gateway["host"],
            )

            logger.info(f"Agent={self.user}@{self.host}:: {file.name} upload done")

    def pull_docker_image(self) -> None:
        """pull georesolver docker image"""
        cmds = docker_pull_cmd()
        _ = ssh_run_cmds(
            cmds,
            user=self.user,
            host=self.host,
            gateway_user=self.gateway["user"],
            gateway_host=self.gateway["host"],
        )

    def agent_start(self) -> None:
        """start georesolver on the remote server"""
        cmd = docker_run_agent_cmd(self.remote_dir, self.agent_config_path)
        pprint(cmd)

        _ = ssh_run_cmd(
            cmd,
            self.user,
            self.host,
            self.gateway["user"],
            self.gateway["host"],
        )

    def run(self) -> None:
        """start docker run on remote server (TODO: locally as well)"""
        # check if remote connection is ok
        self.check_connection()

        # check agent config validity
        self.check_local_dir()

        # create working dir remotely
        self.create_remote_dir()

        # upload target files
        self.upload_target_files()

        # pull docker image
        self.pull_docker_image()

        # start docker container with params
        self.agent_start()
