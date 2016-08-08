# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:07:02 2016

@author: SamWitte
"""

import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
print os.getcwd()
from subhalo import *
import subhalo

a = Observable(100., 3.1 * 10**-26., 'BB', m_low=np.log10(6.48 * 10**6.),
                 m_high=np.log10(2.0 * 10 **9), c_low=np.log10(20.),
                 c_high=2.5, alpha=0.16, profile=0, truncate=True,
                 arxiv_num=10070438)
   
a.Table_Dmax_Pointlike()
print a.N_Pointlike(30.)