# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:52:48 2016

@author: SamWitte
"""

import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'SubhaloDetection')
from subhalo import *


Table_Spatial_Extension(d_low=-3., d_high=1., d_num=80, m_low=np.log10(6.48 * 10**6.), 
                        m_high=np.log10(2.0 * 10 **9), m_num=60, c_low=np.log10(20.),
                        c_high=2.4, c_num = 50, alpha=0.16, profile=0, 
                        truncate=True, arxiv_num=10070438)