from pathlib import Path
from rpy2.robjects.packages import STAP
import rpy2.robjects as robjects


rdir = Path(__file__).parent.parent / 'R_insol_1_2_2'

src = robjects.r['source']
for f in rdir.glob('*.R'):
    src(f.as_posix())

with open(rdir / 'JDymd.R', 'r') as f:
    string = f.read()
R_JDymd = STAP(string, 'JDymd')

with open(rdir / 'sunvector.R', 'r') as f:
    string = f.read()
R_sunvector = STAP(string, 'sunvector')

with open(rdir / 'cgrad.R', 'r') as f:
    string = f.read()
R_cgrad = STAP(string, 'cgrad')

with open(rdir / 'hillshading.R', 'r') as f:
    string = f.read()
R_hillshading = STAP(string, 'hillshading')

with open(rdir / 'sunvector.R', 'r') as f:
    string = f.read()
R_sunvector = STAP(string, 'sunvector')

