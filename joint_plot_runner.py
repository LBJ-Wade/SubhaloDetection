import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from Joint_Sim_Comparison import *


#Joint_Simulation_Comparison().KMMDSM_fit(m_min=1.*10.**4., m_max=1.*10**7., gcd_min=0., gcd_max=100.,
#                                         d_vmax_min=0., d_vmax_max=1., ms=3)

#Joint_Simulation_Comparison().equipartition(x=np.array([0.1, .5, 1., 1.2, 1.4, 1.5, 1.3, 2.2, 0.1, 0.3]), bnd_num=3)


#Joint_Simulation_Comparison().multi_KKDSM_plot(m_num=4, gcd_num=1, del_v_num=1, ms=2)


# LIST: [4.10e+05, 5.35e+06, 1.70e+07, 3.16e+07, 5.52e+07, 1.18e+08, 10.**10.]
#Joint_Simulation_Comparison().fit_hisogram(mlow=4.10e+05, mhigh=5.35e+06)


#Joint_Simulation_Comparison().fit_params()


Joint_Simulation_Comparison().obtain_number_density(min_mass=10.**7.)