"""define the agent class and functions"""

from pathlib import Path
from loguru import logger
from enum import Enum

from geogiant.common.ssh_utils import ssh_run_cmd
from geogiant.common.settings import PathSettings

path_settings = PathSettings()


class ProcessNames(Enum):
    ECS_PROC = "ecs_process"
    SCORE_PROC = "score_process"
    PING_PROC = "ping_process"
    INSERT_PROC = "insert_process"


class Agent:
    def __init__(
        self,
        user: str,
        host: str,
        gateway: dict,
        remote_dir: str,
        experiment_uuid: str,
        processes: dict[dict],
        target_file: Path,
        hostname_file: Path,
        max_ongoing_pings: int,
    ) -> None:
        # remote server ssh parameters
        self.user = user
        self.host = host

        # path for input/output experiment files
        self.remote_dir = remote_dir
        self.experiment_uuid = experiment_uuid

        # georesolver processes and input files
        self.processes = processes
        self.target_file = target_file
        self.hostname_file = hostname_file
        self.max_ongoing_pings = max_ongoing_pings

        # in the case of ssh proxy jump, not nested
        self.gateway = gateway

        # local directories for log and files
        self.experiment_dir = path_settings.EXPERIMENT_PATH / f"{self.experiment_uuid}"
        # use host to define io files
        self.agent_dir = self.experiment_dir / f"{self.host}"

    def check_connection(self) -> None:
        """various check for connection and file existence"""
        # test ssh connection to remote server
        connect_cmd = "echo agent connection ok"
        result = ssh_run_cmd(
            connect_cmd,
            self.user,
            self.host,
            self.gateway["user"],
            self.gateway["host"],
        )

        logger.info(f"Agent={self.user}@{self.host}:: {result.stdout}")

    def check_input_dirs(self) -> None:
        """check that the input file exists"""
        if not self.target_file.exists():
            raise RuntimeError(
                f"Agent={self.user}@{self.host}:: {self.target_file} does not exists"
            )

        if not self.hostname_file.exists():
            raise RuntimeError(
                f"Agent={self.user}@{self.host}:: {self.hostname_file} does not exists"
            )

    def run(self) -> None:
        """start docker run on remote server (TODO: locally as well)"""
        self.check_connection()
        # check agent config validity
        self.check_input_dirs()
        self.check_processes_definition()

        # create working dir locally
        # create working dir remotely

        # upload target files

        # pull docker image

        # start docker container with params
