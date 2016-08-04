"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from Plotting_Functions import *

LimitPlotter(annih_prod='BB', n_obs=0., CL=0.95, pointlike=True,
             alpha=0.16, profile=0, truncate=True, arxiv_num=13131729, b_min=20.,
             mass_low=1., mass_high=3., fs=20).PlotLimit()