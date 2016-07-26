# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:33:11 2016

@author: SamWitte
"""
import os
import matplotlib 
matplotlib.use('agg')
import argparse
import numpy as np
from subhalo import *


parser = argparse.ArgumentParser()
parser.add_argument('--dmax',default=False)
parser.add_argument('--nobs',default=False)
parser.add_argument('--simname',default='sim1')
parser.add_argument('--mass',default=100,type=float)
parser.add_argument('--cross_sec',default=np.log10(3.*10**-26.),type=float)
parser.add_argument('--annih_prod',default='BB')
parser.add_argument('--m_low',default=np.log10(6.48 * 10**6.),type=float)
parser.add_argument('--m_high',default=np.log10(2.0 * 10 **9),type=float)
parser.add_argument('--c_low',default=np.log10(20.),type=float)
parser.add_argument('--c_high',default=2.4,type=float)
parser.add_argument('--alpha',default=0.16,type=float)
parser.add_argument('--profile',default=0,type=int)
parser.add_argument('--truncate',default=True)
parser.add_argument('--arxiv_num',default=10070438,type=int)
parser.add_argument('--m_num',default=30,type=int)
parser.add_argument('--c_num',default=30,type=int)
parser.add_argument('--b_min',default=30.,type=float)
parser.add_argument('--path',default=os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')

args = parser.parse_args()


Build_obs_class = Observable(args.mass, args.cross_sec, args.annih_prod, m_low=args.m_low, 
                             m_high=args.m_high, c_low=args.c_low,
                             c_high=args.c_high, alpha=args.alpha, profile=args.profile, truncate=args.truncate, 
                             arxiv_num=args.arxiv_num)

if args.dmax:                           
    Build_obs_class.Table_Dmax_Extended(m_num=args.m_num, c_num=args.c_num)
    
if args.nobs:
    Build_obs_class.N_Extended(args.b_min, m_num=args.m_num, c_num=args.c_num)

