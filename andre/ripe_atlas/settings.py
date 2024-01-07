"""general tool settings"""
from pathlib import Path

from dotenv import load_dotenv
from common.settings import CommonSettings


class RIPEAtlasSettings(CommonSettings):
    """general settings, credentials"""

    # workaround to override LD_LIBRARY_PATH. Otherwise cannot insert into clickhouse
    load_dotenv(override=True)

    # credentials
    username: str = ""
    secret_key: str = ""
    ip_version: int = 4

    # urls
    base_url: str = "https://atlas.ripe.net/api/v2"
    key_url: str = f"?key={secret_key}"
    measurement_url: str = f"{base_url}/measurements/{key_url}"

    # default ripe atlas parameters
    max_vp: int = 1000
    max_measurement: int = 99
    ping_nb_packets: int = 3
