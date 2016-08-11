import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from Plotting_Functions import *

# plot_profiles(m_sub=1. * 10.**7., arxiv_num=[13131729, 13131729, 160106781, 160106781], density_sqr=True, M200=[False, False, True, True]) #  13131729, 160106781

#Jfactor_plots(m_sub=10.**7., arxiv_num=[13131729], M200=[False], mx=100., cross_sec=3.*10**-26., annih_prod='BB')

extension_vs_dist(m_sub=10.**7., arxiv_num=[13131729, 13131729, 13131729, 13131729], M200=[False, False, False, False])