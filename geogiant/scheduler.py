"""schedule a measurement over a list of agents, split initial target file equally on each agents"""

from pathlib import Path
from random import shuffle
from loguru import logger

from geogiant.agent import ProcessNames

from geogiant.agent import Agent
from geogiant.common.ssh_utils import ssh_run_cmd
from geogiant.common.files_utils import (
    load_csv,
    load_json,
    dump_csv,
    dump_json,
    copy_to,
)
from geogiant.common.settings import PathSettings, RIPEAtlasSettings

path_settings = PathSettings()
ripe_atlas_settings = RIPEAtlasSettings()


def check_input_dirs(target_file: Path, hostname_file: Path) -> None:
    """check that the input file exists"""
    if not target_file.exists():
        raise RuntimeError(f"{target_file} does not exists")

    if not hostname_file.exists():
        raise RuntimeError(f"{hostname_file} does not exists")


def check_processes(process_definition: list[dict]) -> list[str]:
    """check that the required process to run are valid"""
    p_names = []
    for process in process_definition:
        assert "name" in process
        assert "in_table" in process
        assert "out_table" in process
        process_name = process["name"]
        if process_name not in ProcessNames._value2member_map_:
            raise RuntimeError(f"Agent={process} does not suported")

        p_names.append(process_name)

    # check process chain
    if not ProcessNames.ECS_PROC.value in p_names:
        raise RuntimeError(f"Cannot run without {ProcessNames.ECS_PROC.value}")

    if (
        ProcessNames.INSERT_PROC.value in p_names
        and not ProcessNames.PING_PROC.value in p_names
    ):
        raise RuntimeError(
            f"Cannot run {ProcessNames.INSERT_PROC.value} without {ProcessNames.PING_PROC.value}"
        )

    if (
        ProcessNames.PING_PROC.value in p_names
        and not ProcessNames.SCORE_PROC.value in p_names
    ):
        raise RuntimeError(
            f"Cannot run {ProcessNames.PING_PROC.value} without {ProcessNames.SCORE_PROC.value}"
        )

    if (
        ProcessNames.SCORE_PROC.value in p_names
        and not ProcessNames.ECS_PROC.value in p_names
    ):
        raise RuntimeError(
            f"Cannot run {ProcessNames.SCORE_PROC.value} without {ProcessNames.ECS_PROC.value}"
        )


def check_agent_connection(agent_definition: dict) -> None:
    """various check for connection and file existence"""
    # test ssh connection to remote server
    if "host" != "localhost":
        assert "user" in agent_definition

        gateway_user = None
        gateway_host = None
        if "gateway" in agent_definition:
            assert "user" in agent_definition["gateway"]
            assert "host" in agent_definition["gateway"]

            gateway_user = agent_definition["user"]
            gateway_host = agent_definition["host"]

        connect_cmd = "echo agent connection ok"
        result = ssh_run_cmd(
            connect_cmd,
            agent_definition["user"],
            agent_definition["host"],
            gateway_user,
            gateway_host,
        )

    result.stdout = result.stdout.strip("\n")
    logger.info(f"Agent={agent_definition['host']}:: {result.stdout}")


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
    experiment_config["target_file"] = Path(experiment_config["target_file"])
    experiment_config["hostname_file"] = Path(experiment_config["hostname_file"])
    check_input_dirs(
        experiment_config["target_file"],
        experiment_config["hostname_file"],
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


def create_agents(config_path: dict) -> list[Agent]:
    """split experiment over each agents"""
    # check config validity
    config = check_config(config_path)

    # create experiement directory
    experiment_path = create_experiment_path(config["experiment_uuid"])
    # copy hostname file to experiement path (common to all agents)
    copy_to(config["hostname_file"], experiment_path)

    # load targets from target file
    targets = load_csv(config["target_file"])
    shuffle(targets)

    # split measurement load equally among agents
    agents = []
    agent_target_load = len(targets) // len(config["agents"]) + 1
    agent_max_ping = ripe_atlas_settings.MAX_MEASUREMENT // len(config["agents"]) - 1
    for i, agent_definition in enumerate(config["agents"]):
        # create fresh directory for agent
        agent_dir = create_agent_path(agent_definition["host"], experiment_path)

        # upload local agent targets
        agent_targets = targets[i * agent_target_load : (i + 1) * agent_target_load]
        dump_csv(agent_targets, agent_dir / "targets.csv")

        # some parameters are copied from general config
        agent_definition["experiment_uuid"] = config["experiment_uuid"]
        agent_definition["processes"] = config["processes"]
        agent_definition["max_ongoing_ping"] = config["batch_size"]

        # specific agent parameters and files
        agent_definition["max_ongoing_ping"] = agent_max_ping
        agent_definition["local_dir"] = str(agent_dir)
        agent_definition["target_file"] = str(agent_dir / "targets.csv")
        agent_definition["hostname_file"] = str(
            experiment_path / config["hostname_file"].name
        )

        # dump agent config, create and store agent
        dump_json(agent_definition, agent_dir / "config.json")
        agent = Agent(agent_dir / "config.json")
        agents.append(agent)

    return agents


# debugging
if __name__ == "__main__":
    config_path = path_settings.DEFAULT / "../experiment_config/config_example.json"
    agents = create_agents(config_path)
    for agent in agents:
        agent.run()
