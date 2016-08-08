import os, sys
sys.path.insert(0 , os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection')
from Plotting_Functions import *

#  CONTOURS
# plots = model_plots(cross_sec=3.*10**-26., mx=100., annih_prod='BB', pointlike=True,
#                    alpha=0.16, profile=0, truncate=True, arxiv_num=10070438,
#                    mass_low=-5., mass_high=7.)
# plots.plot_flux_contours(threshold=7.*10.**-10, flux_low=-18., flux_high=6.)


#  CLOSER LOOK AT DMAX BEHAVIOR
#  plots = model_plots(cross_sec=10.**-25.482758621, mx=100., annih_prod='BB', pointlike=True,
#                     alpha=0.16, profile=0, truncate=True, arxiv_num=10070438,
#                     mass_low=-5., mass_high=7.)
#
#  plots.dmax_obs_splice()

#  Cross Section vs Nobs
plots = model_plots(cross_sec=3.*10**-26., mx=100., annih_prod='BB', pointlike=True,
                    alpha=0.16, profile=0, truncate=True, arxiv_num=13131729,
                    b_min=20.)
plots.c_sec_v_nobs_plot()
