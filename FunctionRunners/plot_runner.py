"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""
import os, sys
sys.path.insert(0 , os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection')
import numpy as np
from Plotting_Functions import *
from helper import *

#LimitPlotter(annih_prod='BB', n_obs=0., CL=0.95, pointlike=True,
#             alpha=0.16, profile=0, truncate=True, arxiv_num=10070438, b_min=20.,
#             mass_low=1., mass_high=3., fs=20).PlotLimit()

#Multi_Limit_Plots(annih_prod=['BB'], profile=[0,1,0], truncate=[True],
#                  arxiv_num=[13131729, 13131729, 10070438],
#                  b_min=[20.], alpha=0.16, n_obs=[0.], mass_low=1., mass_high=3.,
#                  CL=0.95, fs=20)
plot_spectrum()
integrated_rate_test()