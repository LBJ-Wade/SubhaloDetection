import os, sys
sys.path.insert(0 , os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection')
from subhalo import *

model = DM_Limits(nobs=0., nbkg=0., CL=0.95, annih_prod='BB', pointlike=True,
                  alpha=0.16, profile=0, truncate=True, arxiv_num=13131729, b_min=20.)

model.poisson_limit()