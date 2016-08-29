import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')

from Via_LacteaII_Analysis import *

# Goodness-of-fit
#Via_Lactea_II(profile=0, truncate=True, arxiv_num=13131729).goodness_of_fit()
#Via_Lactea_II(profile=0, truncate=False, arxiv_num=13131729).goodness_of_fit()
#Via_Lactea_II(profile=0, truncate=False, arxiv_num=160106781).goodness_of_fit()
#Via_Lactea_II(profile=1, truncate=True, arxiv_num=13131729).goodness_of_fit()
#Via_Lactea_II(profile=1, truncate=False, arxiv_num=13131729).goodness_of_fit()
#Via_Lactea_II(profile=1, truncate=False, arxiv_num=160106781).goodness_of_fit()



#  Find Subhalos with Mass/GCD property
#Via_Lactea_II().find_subhalos(min_mass=10.**5, max_mass=10. ** 8., gcd_min=0., gcd_max=200.,
#                              del_vmax_min=.7, del_vmax_max=1.)


#ids = Via_Lactea_II().evolution_tracks(return_recent=True)
#Via_Lactea_II().KMMDSM_fit(subhalo_ids=ids)


#  Plot Sample
#plot_sample_comparison(sub_num=2805, plot=True)


#  Catagorical Subplots
#catagorial_subplots(min_mass=1 * 10 ** 3., max_mass=5 * 10. ** 8., gcd_min=50., gcd_max=100.,
#                    n_plots=4, tag='_50to100_kpc_GC')


#  Mass Ratio Analysis
#mass_ratio_vs_gcd(min_mass=1.*10.**4., max_mass=1.*10**8., var1='M300', var2='Mtot')


#  Number Suhalo Histogram
#number_subhalo_histogram(mass_low=10. ** 4, mass_high = 10.**8., gcd_low=0., gcd_high=4000., nbins=20)

#  Uncertainty file
#Via_Lactea_II().Density_Uncertanity()


#  [m_low_min = 4.1 * 10 ** 3., m_high_max = 10.**12., gc_d_max (max) = 4000]
#plot_slice(m_low=4.1 * 10 ** 3., m_high =10.**12., gc_d_min=0., gc_d_max=2000., plot=True, alpha=.3, ms=3)


#multi_slice(m_low=4.1 * 10 ** 3., m_high =10.**7., m_num=2, gc_d_min=0., gc_d_max=4000.,
#            gc_d_num=5, plot=True, alpha=.4, ms=1, p_mtidal=False, p_mrmax=True,
#            p_m300=False, p_m600=False)


#Inner Outer Slopes
#Preferred_Density_Slopes_VL(mass_low=1*10.** 4, mass_high = 1*10.**10., gcd_min=0., gcd_max=20., tag='0to20kpc')

Via_Lactea_II().obtain_number_density()

#Via_Lactea_II().einasto_density_fit()