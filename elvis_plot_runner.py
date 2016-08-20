import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from ELVIS_Analysis import *


#  Find Subhalos with Mass/GCD property
#ELVIS().find_subhalos(min_mass=10.**5, max_mass=10. ** 7., gcd_min=0., gcd_max=20.,
#                      del_vmax_min=0., del_vmax_max=1.)


#  Plot Sample
plot_sample_comparison_elv(sub_num=22071, plot=True)


#  Catagorical Subplots
#catagorial_subplots_elv(min_mass=1 * 10 ** 3., max_mass=5 * 10. ** 8., gcd_min=50., gcd_max=100.,
#                    n_plots=4, tag='_50to100_kpc_GC')


#  Mass Ratio Analysis
#mass_ratio_vs_gcd_elv(min_mass=1.*10.**4., max_mass=1.*10**8., var1='M300', var2='Mtot')


#  Number Suhalo Histogram
#number_subhalo_histogram_elv(mass_low=10. ** 4, mass_high = 10.**8., gcd_low=0., gcd_high=4000., nbins=20)

#  Uncertainty file
#ELVIS().Density_Uncertanity()


#  [m_low_min = 4.1 * 10 ** 3., m_high_max = 10.**12., gc_d_max (max) = 4000]
#plot_slice_elv(m_low=4.1 * 10 ** 3., m_high =10.**12., gc_d_min=0., gc_d_max=2000., plot=True, alpha=.3, ms=3)


#multi_slice_elv(m_low=10**5., m_high =10.**7., m_num=2, gc_d_min=0., gc_d_max=100.,
#                gc_d_num=5, plot=True, alpha=.4, ms=1, p_mtidal=False, p_mrmax=True)


#Inner Outer Slopes
#Preferred_Density_Slopes_elv(mass_low=1*10.** 4, mass_high = 1*10.**7., gcd_min=0., gcd_max=20., tag='0to20kpc')
