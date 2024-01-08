from pathlib import Path
from common.settings import CommonSettings 
  
class ZDNSSettings(CommonSettings):
    """ZNDS module settings"""

    # ZDNS binary path
    DEFAULT_PATH: Path = Path(__file__).resolve().parent
    EXEC_PATH: Path = DEFAULT_PATH / "zdns_binary"
    
    # ZDNS tool parameters
    RECORD_TYPE: str = "A"
    