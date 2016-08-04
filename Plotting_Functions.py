# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from helper import *
from subhalo import *
import os
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.use('pdf')
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'serif','serif':['Times','Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size']=8
mpl.rcParams['ytick.major.size']=8
mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18


class LimitPlotter(object):

    def __init__(self, annih_prod='BB', n_obs=0., CL=0.95, pointlike=True,
                 alpha=0.16, profile=0, truncate=True, arxiv_num=13131729, b_min=20.,
                 mass_low=1., mass_high=3., fs=20):

        Profile_List = ["Einasto", "NFW"]
        self.profile_name = Profile_List[profile]
        self.annih_prod = annih_prod
        self.nobs = n_obs
        self.CL = CL
        self.pointlike = pointlike
        self.alpha = alpha
        self.profile = profile
        self.truncate = truncate
        self.arxiv_num = arxiv_num
        self.b_min = b_min
        self.mass_low = mass_low
        self.mass_high = mass_high
        self.fs = fs
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"

    def check_if_lim_file_exists(self):

        if self.pointlike:
            plike_tag = '_Pointlike'
        else:
            plike_tag = '_Extended'

        self. file_name = 'Limits' + plike_tag + self.profile_name + '_Truncate_' + \
                           str(self.truncate) + '_alpha_' + str(self.alpha) +\
                          '_annih_prod_' + self.annih_prod + '_arxiv_num_' +\
                          str(self.arxiv_num) + '_bmin_' + str(self.b_min) + '.dat'

        if os.path.exists(self.folder + self.file_name):
            return True
        else:
            return False

    def PlotLimit(self):

        if not self.check_if_lim_file_exists():
            model = DM_Limits(nobs=self.nobs, CL=self.CL, annih_prod=self.annih_prod,
                              pointlike=self.pointlike, alpha=self.alpha, profile=self.profile,
                              truncate=self.truncate, arxiv_num=self.arxiv_num, b_min=self.b_min)

            model.poisson_limit()

        color_list = ['Black', 'Red', 'Blue', 'Purple']
        lim_file = np.loadtxt(self.folder + self.file_name)

        lim_interp = interp1d(lim_file[:,0], lim_file[:,1], kind='cubic', bounds_error=False)

        mass_tab = np.logspace(self.mass_low, self.mass_high, 300)
        # limpt = lim_interp(mass_tab)
        limpt = interpola(mass_tab, lim_file[:,0], lim_file[:,1])

        fig = plt.figure()
        ax = plt.gca()
        if self.annih_prod == 'BB':
            label = r'$\chi \chi \rightarrow b \bar{b}$'
        else:
            label = ''

        plt.plot(mass_tab, limpt, lw=2, color=color_list[0], label=label)
        if self.truncate:
            trunclabel = 'Truncated'
        else:
            trunclabel = ''
        if self.pointlike:
            plabel = 'Pointlike'
        else:
            plabel = 'Extended'

        ax.set_xscale("log")
        ax.set_yscale('log')

        plt.title('{}'.format(self.profile_name), fontsize=20)

        plt.text(15., 4. * 10**-24., '{}\n{}'.format(plabel, trunclabel), fontsize=15, ha='left', va='center')
        plt.text(900., 2. * 10 ** -24.,label, fontsize=15, ha='right')

        pl.xlim([10., 1000.])
        pl.ylim([10**-27., 10**-23.])
        pl.ylabel(r'$\left< \sigma v \right>$   [$cm^3/s$] ', fontsize=self.fs)
        pl.xlabel(r'$m_{\chi}$  [$GeV$]', fontsize=self.fs)
        figname = self.folder + '../Plots/' + 'Subhalo_Limits_{}_{}_'.format(self.profile_name, self.truncate) +\
                  '{}_AnnihProd_{}_Cparams_{}_'.format(self.pointlike, self.annih_prod, self.arxiv_num) +\
                  'bmin_{}_CL_{}_Nobs_{}.pdf'.format(self.b_min,self.CL,self.nobs)

        fig.set_tight_layout(True)

        pl.savefig(figname)


class dmax_plots(object):

    def __init__(self, cross_sec=3.*10**-26., mx=100., annih_prod='BB', pointlike=True, alpha=0.16, profile=0,
                 truncate=True, arxiv_num=10070438, mass_low=-5., mass_high=7., fs=20):
        self.cross_sec = cross_sec
        self.mx = mx
        self.annih_prod = annih_prod
        self.pointlike = pointlike
        self.alpha = alpha
        self.profile = profile
        self.truncate = truncate
        self.arxiv_num = arxiv_num
        self.mass_low = mass_low
        self.mass_high = mass_high
        self.fs = fs
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"

    def constant_flux_contours(self, threshold=10**-9., flux_low=-18., flux_high=6.):

        fluxtab = np.power(10.,np.linspace(flux_low, flux_high, int(flux_high - flux_low)+1))
        mass_tab = np.logspace(self.mass_low, self.mass_high, 100)
        mass_dist_tab = np.zeros(mass_tab.size * fluxtab.size).reshape((mass_tab.size, fluxtab.size))
        real_dmax = np.zeros(mass_tab.size)

        for i,m in enumerate(mass_tab):
            if self.truncate:
                mm = m / 0.005
            else:
                mm = m
            model = Model(self.mx, self.cross_sec, self.annih_prod, mm, self.alpha,
                          truncate=self.truncate, arxiv_num=self.arxiv_num,
                          profile=self.profile, point_like=self.pointlike)

            real_dmax[i] = model.d_max_point(threshold=threshold)
            #  extend_dmax[i] = model.D_max_extend()

            for j,f in enumerate(fluxtab):
                mass_dist_tab[i,j] = model.d_max_point(threshold=f)

        return real_dmax, mass_dist_tab

    def plot_flux_contours(self, threshold=10**-9., flux_low=-18., flux_high=6.):
        # TODO: Find way to determine spatial extension boundary
        dmax, contours = self.constant_flux_contours(threshold=threshold,
                                                               flux_low=flux_low,
                                                               flux_high=flux_high)

        mass_tab = np.logspace(self.mass_low, self.mass_high, 100)
        mlen,flen = np.shape(contours)

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()

        plt.plot(mass_tab, dmax, lw=2, color='Black')
        #  plt.plot(mass_tab, dmax_ext, lw=2, color='Black')
        for i in range(flen):
            plt.plot(mass_tab, contours[:,i], lw=1, color='Black', alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10**self.mass_low, 10**self.mass_high])
        pl.ylim([10**-4., 10.])
        Profile_List = ["Einasto","NFW"]
        profile_name = Profile_List[self.profile]
        plt.title('{}'.format(profile_name), fontsize=20)
        if self.pointlike:
            plabel = 'Pointlike'
        else:
            plabel = 'Extended'
        if self.truncate:
            trunclabel = 'Truncated'
        else:
            trunclabel = ''

        plt.text(2. * 10**-5, 1., '{}\n{}\n $m_\chi = ${} \n $\sigma v = ${}'.format(plabel,trunclabel,
                                                                                  self.mx, self.cross_sec),
                 fontsize=15, ha='left', va='center')

        pl.ylabel('Distance  [kpc]', fontsize=self.fs)
        pl.xlabel(r'$M_{subhalo}$  [$M_\odot$]', fontsize=self.fs)

        figname = self.folder + '../Plots/' + 'DistanceCountours_{}_{}_'.format(profile_name, trunclabel) + \
                  '{}_AnnihProd_{}_CrossSection_{}'.format(plabel, self.annih_prod, self.cross_sec) + \
                  '_Cparams_{}_.pdf'.format(self.arxiv_num)

        #  adjustFigAspect(fig,aspect=.5)
        fig.set_tight_layout(True)
        pl.savefig(figname)


