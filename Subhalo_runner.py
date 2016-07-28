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

profile_list = ['Einasto', 'NFW']
pf = profile_list[args.profile]
simga_n_file = pf + '_Truncate_' + args.truncate + '_Cparam_' + args.arxiv_num +\
               '_alpha_' + args.alpha + '_mx_' + args.mass + '_annih_prod_' +\
               args.annih_prod + '_bmin_' + str(args.b_min) + '.dat'

Build_obs_class = Observable(args.mass, args.cross_sec, args.annih_prod, m_low=args.m_low, 
                             m_high=args.m_high, c_low=args.c_low,
                             c_high=args.c_high, alpha=args.alpha, profile=args.profile, truncate=args.truncate, 
                             arxiv_num=args.arxiv_num)

if args.dmax:                           
    Build_obs_class.Table_Dmax_Extended(m_num=args.m_num, c_num=args.c_num)
    
if args.nobs:
    n_ext_obs = Build_obs_class.N_Extended(args.b_min, m_num=args.m_num, c_num=args.c_num)
    if os.path.isfile(args.path + '/Data/' + simga_n_file):
        cross_sec_nobs = np.loadtxt(args.path + '/Data/' + simga_n_file)
        add_to_table = np.vstack((cross_sec_nobs,[args.cross_sec, n_ext_obs]))
        np.savetxt(args.path + '/Data/' + simga_n_file, add_to_table)
    else:
         np.savetxt(args.path + '/Data/' + simga_n_file, np.array([args.cross_sec, n_ext_obs]))

