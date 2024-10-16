"""methods for plotting graph"""

import numpy as np

from pathlib import Path
from loguru import logger
from collections import defaultdict, OrderedDict

from geogiant.evaluation.plot import ecdf
from geogiant.common.files_utils import load_pickle
from geogiant.common.utils import EvalResults
from geogiant.common.settings import PathSettings

path_settings = PathSettings()






