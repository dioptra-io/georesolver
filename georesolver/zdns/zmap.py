import subprocess

from random import shuffle
from loguru import logger
from pathlib import Path

from georesolver.common.files_utils import (
    load_csv,
    create_tmp_csv_file,
)
from georesolver.common.settings import PathSettings, ClickhouseSettings

path_settings = PathSettings()
clickhouse_settings = ClickhouseSettings()


def get_zmap_cmd(input_file: Path, output_file: Path) -> str:
    return f"echo {path_settings.SUDO_PWD} | sudo -S zmap --probe-module=icmp_echoscan -p 0 -r 10000 --target-file={input_file}  -O csv --output-file={output_file}"


def zmap(subnets: list[str]) -> None:
    """probe a list of IP addresses from a list of subnets"""
    input_file = create_tmp_csv_file([str(subnet) + "/24" for subnet in subnets])
    output_file = create_tmp_csv_file([])
    zmap_cmd = get_zmap_cmd(input_file, output_file)

    results = subprocess.run(
        args=zmap_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )

    if results.stderr:
        pass

    responsive_ip_address = load_csv(output_file)

    logger.debug(
        f"Responsive IP addresses for {len(subnets)}:: {len(responsive_ip_address)}"
    )

    input_file.unlink()
    output_file.unlink()

    return responsive_ip_address
