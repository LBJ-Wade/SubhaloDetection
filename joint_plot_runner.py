import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from Joint_Sim_Comparison import *


Joint_Simulation_Comparison().KMMDSM_fit(m_min=10.**4., m_max=10**10., gcd_min=0., gcd_max=300.,
                                         d_vmax_min=0.0, d_vmax_max=0.05, ms=3)
