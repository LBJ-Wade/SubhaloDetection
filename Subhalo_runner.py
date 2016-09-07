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
parser.add_argument('--dmax', default=False)
parser.add_argument('--nobs', default=False)
parser.add_argument('--simname', default='sim1')
parser.add_argument('--tag', default='')
parser.add_argument('--pointlike', default=False)
parser.add_argument('--mass', default=40, type=float)
parser.add_argument('--cross_sec', default=np.log10(3.*10**-26.), type=float)
parser.add_argument('--annih_prod', default='BB', type=str)
parser.add_argument('--m_low', default=np.log10(3.24 * 10.**4.), type=float)
parser.add_argument('--m_high', default=np.log10(1.0 * 10.**7.), type=float)
parser.add_argument('--c_low', default=np.log10(5.), type=float)
parser.add_argument('--c_high', default=2., type=float)
parser.add_argument('--alpha', default=0.16, type=float)
parser.add_argument('--profile', default=1, type=int)
parser.add_argument('--truncate', default=False)
parser.add_argument('--arxiv_num', default=10070438, type=int)
parser.add_argument('--b_min', default=20., type=float)
parser.add_argument('--m_num', default=25, type=int)
parser.add_argument('--c_num', default=15, type=int)
parser.add_argument('--thresh', default=7 * 10.**-10., type=float)
parser.add_argument('--M200', default=False)
parser.add_argument('--gamma', default=0.945, type=float)
parser.add_argument('--stiff_rb', default=False)
parser.add_argument('--path', default=os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection/')

args = parser.parse_args()


def str2bool(v):
    if type(v) == bool:
        return v
    elif type(v) == str:
        return v.lower() in ("yes", "true", "t", "1")

dmax = str2bool(args.dmax)
nobs = str2bool(args.nobs)
pointlike = str2bool(args.pointlike)
truncate = str2bool(args.truncate)
m200 = str2bool(args.M200)
stiff_rb = str2bool(args.stiff_rb)

Profile_list = ["Einasto", "NFW", "HW"]
pf = Profile_list[args.profile]
if pointlike:
    plike_tag = '_Pointlike'
else:
    plike_tag = '_Extended'

if args.profile < 2:
    extra_tag = '_Truncate_' + str(args.truncate) + '_Cparam_' + str(args.arxiv_num) +\
               '_alpha_' + str(args.alpha)
else:
    extra_tag = '_Gamma_{:.3f}_Stiff_rb_'.format(args.gamma) + str(stiff_rb)

simga_n_file = pf + '_mx_' + str(args.mass) + '_annih_prod_' + args.annih_prod + '_bmin_' +\
               str(args.b_min) + plike_tag + extra_tag + '_Mlow_{:.3f}'.format(args.m_low) +\
               args.tag + '.dat'

nobs_dir = "/Cross_v_Nobs/"

Build_obs_class = Observable(args.mass, args.cross_sec, args.annih_prod, m_low=args.m_low, 
                             m_high=args.m_high, c_low=args.c_low,
                             c_high=args.c_high, alpha=args.alpha, profile=args.profile, truncate=truncate,
                             arxiv_num=args.arxiv_num, point_like=args.pointlike, gam=args.gamma,
                             stiff_rb=stiff_rb, m200=m200)

if dmax:
    if pointlike:
        Build_obs_class.Table_Dmax(m_num=args.m_num, c_num=args.c_num,
                                             threshold=args.thresh)
    else:
        Build_obs_class.Table_Dmax(m_num=args.m_num, c_num=args.c_num, plike=pointlike)
    
if nobs:
    if pointlike:
        n_point_obs = Build_obs_class.N_obs(args.b_min)
        if os.path.isfile(args.path + '/Data/' + nobs_dir + simga_n_file):
            cross_sec_nobs = np.loadtxt(args.path + '/Data/' + nobs_dir + simga_n_file)
            add_to_table = np.vstack((cross_sec_nobs, [args.cross_sec, n_point_obs]))
            save_tab = add_to_table[np.lexsort(np.fliplr(add_to_table).T)]
            np.savetxt(args.path + '/Data/' + nobs_dir + simga_n_file, save_tab)
        else:
            np.savetxt(args.path + '/Data/' + nobs_dir + simga_n_file, np.array([args.cross_sec, n_point_obs]))
    else:
        n_ext_obs = Build_obs_class.N_obs(args.b_min, plike=pointlike)
        if os.path.isfile(args.path + '/Data/' + nobs_dir + simga_n_file):
            cross_sec_nobs = np.loadtxt(args.path + '/Data/' + nobs_dir + simga_n_file)
            add_to_table = np.vstack((cross_sec_nobs, [args.cross_sec, n_ext_obs]))
            save_tab = add_to_table[np.lexsort(np.fliplr(add_to_table).T)]
            np.savetxt(args.path + '/Data/' + nobs_dir + simga_n_file, save_tab)
        else:
            np.savetxt(args.path + '/Data/' + nobs_dir + simga_n_file, np.array([args.cross_sec, n_ext_obs]))
