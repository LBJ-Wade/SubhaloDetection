#!/usr/bin/env python
import subprocess as sp
import os,sys,fnmatch
import argparse
import pickle
import numpy as np

import subhalo

parser = argparse.ArgumentParser()
parser.add_argument('--dmax', default=True)
parser.add_argument('--nobs', default=True)
parser.add_argument('--tag', default='')
parser.add_argument('--mass', default=1000, type=float)
parser.add_argument('--pointlike', default=True)
parser.add_argument('--cross_sec_low', default=-27., type=float)  # In log10
parser.add_argument('--cross_sec_high', default=-23., type=float)  # In log10
parser.add_argument('--annih_prod', default='BB')  # TODO: add more than b-bbar
parser.add_argument('--m_low', default=np.log10(6.48 * 10**6.), type=float)  # In log10
parser.add_argument('--m_high', default=np.log10(2.0 * 10 **9), type=float)  # In log10
parser.add_argument('--c_low', default=np.log10(2.5), type=float)  # In log10
parser.add_argument('--c_high', default=2.4, type=float)  # In log10
parser.add_argument('--alpha', default=0.16, type=float)  # For Einasto
parser.add_argument('--profile', default=0, type=int)  # [Einasto, NFW] 0 -- 1
parser.add_argument('--truncate', default=True)
parser.add_argument('--arxiv_num', default=13131729, type=int) # [10070438, 13131729]
parser.add_argument('--b_min', default=20., type=float)
parser.add_argument('--m_num', default=20, type=int)
parser.add_argument('--c_num', default=20, type=int)
parser.add_argument('--n_runs', type=int, default=30)
parser.add_argument('--path', default=os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection/')

args = parser.parse_args()

tag = args.tag
mass = args.mass
plike = args.pointlike
cross_sec_low = args.cross_sec_low
cross_sec_high = args.cross_sec_high
annih_prod = args.annih_prod
m_low = args.m_low
m_high = args.m_high
c_low = args.c_low
c_high = args.c_high
alpha = args.alpha
profile = args.profile
arxiv_num = args.arxiv_num
m_num = args.m_num
c_num = args.c_num
n_runs = args.n_runs
path = args.path
truncate = args.truncate
b_min = args.b_min
dmax = args.dmax
nobs = args.nobs

cross_sec_list = np.logspace(cross_sec_low, cross_sec_high, n_runs)

cmds = []
count = 0

for i,sv in enumerate(cross_sec_list):
    simname='sim%d' % i
    cmd = 'cd '+ path + '\n' + 'python Subhalo_runner.py --simname {} --pointlike {} '.format(simname, plike) +\
                               '--mass {} --cross_sec {:.8e} --annih_prod {} '.format(mass, sv, annih_prod) +\
                               '--m_low {:.16f} --m_high {:.16f} --c_low {} '.format(m_low, m_high, c_low) +\
                               '--c_high {} --alpha {} --profile {} '.format(c_high, alpha, profile) +\
                               '--arxiv_num {} --m_num {} --c_num {} '.format(arxiv_num, m_num, c_num) +\
                               '--truncate {} --dmax {}'.format(truncate, dmax)
    cmds.append(cmd)
    count += 1
    
for i in range(count):
    fout=open('runs_dmax/calc_Dmax_{}_{}.sh'.format(tag, i+1), 'w')
    for cmd in cmds[i::count]:
        fout.write('{}\n'.format(cmd))
    fout.close()

fout = open('runs_dmax/Calc_Dmax_commandrunner_{}.sh'.format(tag), 'w')
fout.write('#! /bin/bash\n')
fout.write('#$ -l h_rt=24:00:00,h_data=2G\n')
fout.write('#$ -cwd\n')
fout.write('#$ -t 1-{}\n'.format(count))
fout.write('#$ -V\n')
fout.write('bash calc_Dmax_{}_$SGE_TASK_ID.sh\n'.format(tag))
fout.close()

cmds = []
count = 0

for i,sv in enumerate(cross_sec_list):
    simname='sim%d' % i
    cmd = 'cd '+ path + '\n' + 'python Subhalo_runner.py --simname {} --pointlike {} '.format(simname, plike) +\
                               '--mass {} --cross_sec {:.8e} --annih_prod {} '.format(mass, sv, annih_prod) +\
                               '--m_low {:.16f} --m_high {:.16f} --c_low {} '.format(m_low, m_high, c_low) +\
                               '--c_high {} --alpha {} --profile {} '.format(c_high, alpha, profile) +\
                               '--arxiv_num {} --m_num {} --c_num {} '.format(arxiv_num, m_num, c_num) +\
                               '--truncate {} --b_min {} --nobs {}'.format(truncate, b_min, nobs)
    cmds.append(cmd)
    count += 1
    
for i in range(count):
    fout=open('runs_dmax/calc_Nobs_{}_{}.sh'.format(tag, i+1), 'w')
    for cmd in cmds[i::count]:
        fout.write('{}\n'.format(cmd))
    fout.close()

fout = open('runs_dmax/Nobs_commandrunner_{}.sh'.format(tag), 'w')
fout.write('#! /bin/bash\n')
fout.write('#$ -l h_rt=24:00:00,h_data=2G\n')
fout.write('#$ -cwd\n')
fout.write('#$ -t 1-{}\n'.format(count))
fout.write('#$ -V\n')
fout.write('bash calc_Nobs_{}_$SGE_TASK_ID.sh\n'.format(tag))
fout.close()

