# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:51:20 2016

@author: SamWitte
"""

import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'SubhaloDetection')
from subhalo import *
import numpy as np

a = Observable(100., 3. * 10**-26., 'BB', m_low=np.log10(6.48 * 10**6.), 
               m_high=np.log10(2.0 * 10 **9), c_low=np.log10(20.),
               c_high=2.4, alpha=0.16, profile=0, truncate=True, 
               arxiv_num=10070438)

a.Table_Dmax_Extended(m_num=80, c_num = 50)