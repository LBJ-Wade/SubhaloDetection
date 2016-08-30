# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:52:48 2016

@author: SamWitte
"""

import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from halo_information import *

table_spatial_extension(profile=0, truncate=True, arxiv_num=13131729,
                        M200=False, d_low=-3., d_high=1., d_num=30, 
                        m_num=20, c_num=20)
