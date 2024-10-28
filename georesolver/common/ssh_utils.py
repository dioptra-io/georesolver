"""functions utilities for excuting shell commands on remote server with fabric"""

from pathlib import Path
from loguru import logger
from fabric import Connection, Result

from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def check_error(result: Result, exit_on_failure: bool = False) -> None:
    """check if error in cmd result"""
    if result.stderr:
        if exit_on_failure:
            raise RuntimeError(f"{result.command} execution failed:: {result.stderr}")
        else:
            logger.error(f"{result.command} execution failed:: {result.stderr}")


def ssh_run_cmd(
    cmd: str,
    user: str,
    host: str,
    gateway_user: str = None,
    gateway_host: str = None,
    hide: bool = True,
    exit_on_failure: bool = False,
) -> Result:
    """run a single command on a remote server via ssh, gatewaying with ProxyJump but not nested"""
    if not gateway_user or not gateway_host:
        with Connection(host=host, user=user) as session:
            result: Result = session.run(cmd, hide=hide)

    else:
        with Connection(
            host=gateway_host,
            user=gateway_user,
            gateway=Connection(
                host=host,
                user=user,
            ),
        ) as session:
            result = session.run(cmd, hide=hide)

    check_error(result, exit_on_failure)

    return result


def ssh_run_cmds(
    cmds: list[str],
    user: str,
    host: str,
    gateway_user: str = None,
    gateway_host: str = None,
    hide: bool = True,
    exit_on_failure: bool = False,
) -> list[Result]:
    """same as ssh_run_cmd but for a list of cmd"""
    results = []  # list of cmd results
    if not gateway_user or not gateway_host:
        with Connection(host=host, user=user) as session:
            for cmd in cmds:
                result = session.run(cmd, hide=hide)
                check_error(result, exit_on_failure)
                results.append(result)

    else:
        with Connection(
            host=gateway_host,
            user=gateway_user,
            gateway=Connection(
                host=host,
                user=user,
            ),
        ) as session:
            for cmd in cmds:
                result = session.run(cmd, hide=hide)
                check_error(result, exit_on_failure)
                results.append(result)

    return results


def ssh_upload_file(
    in_file: Path,
    out_file: Path,
    user: str,
    host: str,
    gateway_user: str = None,
    gateway_host: str = None,
) -> Result:

    # upload hostname file
    if not gateway_user or not gateway_host:
        with Connection(host=host, user=user) as session:
            result = session.put(
                local=str(in_file),
                remote=str(out_file),
            )

    else:
        with Connection(
            host=gateway_host,
            user=gateway_user,
            gateway=Connection(
                host=host,
                user=user,
            ),
        ) as session:
            result = session.put(
                local=str(in_file),
                remote=str(out_file),
            )

    return result
