# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 2016

@author: SamWitte
"""
import numpy as np
import os
from subhalo import *
from profiles import NFW, Einasto
import helper
import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import warnings
import glob
from Via_LacteaII_Analysis import *
from ELVIS_Analysis import *

warnings.filterwarnings('error')

#  mpl.use('pdf')
rc('font', **{'family': 'serif', 'serif': ['Times', 'Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

try:
    MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'


class Joint_Simulation_Comparison(object):

    def __init__(self):
        self.dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
        self.sim_list = ['ViaLacteaII_Useable_Subhalos.dat', 'Elvis_Useable_Subhalos.dat']
        self.class_list = [Via_Lactea_II(), ELVIS()]
        self.subhalos = [np.loadtxt(self.dir + self.sim_list[0]),
                         np.loadtxt(self.dir + self.sim_list[1])]
        self.names = ['Via Lactea II', 'ELVIS']

    def KMMDSM_fit(self, m_min=10.**4., m_max=10**10., gcd_min=0., gcd_max=2000.,
                    d_vmax_min=0., d_vmax_max=1., ms=4):
        color_list = ['Red', 'Blue']
        shape_list = ['o', '+']
        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        pl.xlim([10 ** -2, 2.])
        pl.ylim([-0.02, 2.])

        fail_count = 0.
        tot_count = 0.
        for i in range(len(self.class_list)):
            print self.names[i]
            sub_o_int = self.class_list[i].find_subhalos(min_mass=m_min, max_mass=m_max,
                                                         gcd_min=gcd_min,
                                                         gcd_max=gcd_max,
                                                         del_vmax_min=d_vmax_min,
                                                         del_vmax_max=d_vmax_max,
                                                         print_info=False)

            for j, sub in enumerate(self.subhalos[i][sub_o_int]):
                rvmax, vmax, mass = [sub[4], sub[3], sub[5]]
                try:
                    bf_gamma = find_gamma(mass, vmax, rvmax)
                except ValueError:
                    fail_count += 1
                    bf_gamma = 0.0
                subhalo_kmmdsm = KMMDSM(mass, bf_gamma, arxiv_num=160106781,
                                        vmax=vmax, rmax=rvmax)
                plt.plot(subhalo_kmmdsm.rb, bf_gamma, shape_list[i], ms=ms, color=color_list[i])
                tot_count += 1

        fail_frac = float(fail_count / tot_count)
        print 'Fail Fraction: ', fail_count, tot_count
        plt.text(.012, 1.5, 'VLII', fontsize=10, ha='left', va='center', color=color_list[0])
        plt.text(.012, 1.4, 'ELVIS', fontsize=10, ha='left', va='center', color=color_list[1])
        plt.text(.012, 1.3, r'Mass Range [{:.2e} {:.2e}]'.format(m_min, m_max),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(.012, 1.2, r'GC Dist [{:.1f} {:.1f}]'.format(gcd_min, gcd_max),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(.012, 1.1, r'$\Delta$ Vmax [{:.2f} {:.2f}]'.format(d_vmax_min, d_vmax_max),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(.012, 1., 'Fraction Failures: {:.3f}'.format(fail_frac),
                 fontsize=10, ha='left', va='center', color='Black')

        fig_name = self.dir + 'Joint_Sim_Plots/' +\
            'KMMDSM_Fit_mass_range_[{:.2e} {:.2e}]'.format(m_min, m_max) +\
            '_GCD_range_[{:.1f} {:.1f}]'.format(gcd_min, gcd_max) +\
            '_DeltaVmax_[{:.2f} {:.2f}]'.format(d_vmax_min, d_vmax_max) + '.pdf'

        pl.xlabel(r'$R_b$   [kpc]', fontsize=20)
        pl.ylabel(r'$\gamma$', fontsize=20)
        fig.set_tight_layout(True)
        pl.savefig(fig_name)
        return
