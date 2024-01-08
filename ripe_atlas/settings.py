"""general tool settings"""
from pathlib import Path

from dotenv import load_dotenv
from common.settings import CommonSettings

class RIPEAtlasSettings(CommonSettings):
    """general settings, credentials"""

    # workaround to override LD_LIBRARY_PATH. Otherwise cannot insert into clickhouse
    load_dotenv(override=True)

    # credentials
    USERNAME: str = ""
    SECRET_KEY: str = ""
    IP_VERSION: int = 4

    # urls
    BASE_URL: str = "https://atlas.ripe.net/api/v2"
    KEY_URL: str = f"?key={SECRET_KEY}"
    MEASUREMENT_URL: str = f"{BASE_URL}/measurements/{KEY_URL}"

    # default ripe atlas parameters
    MAX_VP: int = 1000
    MAX_MEASUREMENT: int = 99
    PING_NB_PACKETS: int = 3
