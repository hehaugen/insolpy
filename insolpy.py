import rpy2.robjects as robjects
from pathlib import Path
from rpy2.robjects.packages import STAP
import os

import config

res = config.R_JDymd.JDymd(2000,1,1,0,0,0)

sv = config.R_sunvector.sunvector(res[0], 45.135375, -106.775072, -7)
