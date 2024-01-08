"""general tool settings"""
from pathlib import Path

from common.settings import CommonSettings


class ZDNSSettings(CommonSettings):
    """ZNDS settings"""

    # ZDNS binary path
    DEFAULT_PATH: Path = Path(__file__).resolve().parent
    EXEC : Path = DEFAULT_PATH / "zdns"

