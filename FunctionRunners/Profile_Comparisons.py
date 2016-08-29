import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from Plotting_Functions import *

#plot_profiles(m_sub=1.*10.**7., density_sqr=False)

#Jfactor_plots(m_sub=10.**7., dist=1.)

extension_vs_dist(m_sub=10.**7.)