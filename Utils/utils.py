import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from Kernels.kernel_function import PortfolioKernel
from download_JKP.read_dataset import (
    available_jkp_characteristics,
    available_jkp_years,
    load_jkp_value,
)

