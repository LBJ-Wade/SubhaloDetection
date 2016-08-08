# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from helper import *
from subhalo import *
import os
from scipy.interpolate import interp1d, interp2d
import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools

#  mpl.use('pdf')
rc('font', **{'family': 'serif', 'serif': ['Times', 'Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18


def Multi_Limit_Plots(annih_prod=['BB', 'BB'], profile=[0, 1], truncate=[True, True],
                      arxiv_num=[13131729, 13131729], b_min=[20., 20.],
                      alpha=0.16, n_obs=[0., 0.], pointlike=[True, True],
                      mass_low=1., mass_high=3., CL=0.95, fs=20, save_fig=True,
                      color_list=['Black', 'Red', 'Blue', 'Orange', 'Magenta', 'Aqua']):

    """
    Wrapper function that calls LimitPlotter for each list index
    """
    profile_list = ['Einasto', 'NFW']
    lap = len(annih_prod)
    lpro = len(profile)
    ltrun = len(truncate)
    larx = len(arxiv_num)
    lbmin = len(b_min)
    lnobs = len(n_obs)
    lenplike = len(pointlike)
    length_list = [lap, lpro, ltrun, larx, lbmin, lnobs, lenplike]
    list_prop = [annih_prod, profile, truncate, arxiv_num, b_min, n_obs, pointlike]
    max_length = max(length_list)
    if max_length == 1:
        print 'Why are you using this function? Exiting...'
        exit()
    else:
        for i in range(len(length_list)):
            hold = list_prop[i]
            while length_list[i] < max_length:
                hold = [hold, list_prop[i]]
                hold = list(itertools.chain.from_iterable(hold))
                length_list[i] = len(hold)
            list_prop[i] = hold
    print list_prop
    figname = MAIN_PATH + '/SubhaloDetection/Plots/MultiLimitPlot_'

    mt = np.zeros(300 * max_length).reshape((300, max_length))
    lim = np.zeros(300 * max_length).reshape((300, max_length))
    labels = []
    if pointlike:
        tlab = 'T'
    else:
        tlab = 'NT'
    for i in range(max_length):
        mt[:, i], lim[:, i] = LimitPlotter(annih_prod=list_prop[0][i], n_obs=list_prop[5][i], CL=CL,
                                           pointlike=list_prop[6][i], alpha=alpha, profile=list_prop[1][i],
                                           truncate=list_prop[2][i], arxiv_num=list_prop[3][i],
                                           b_min=list_prop[4][i],
                                           mass_low=mass_low, mass_high=mass_high).PlotLimit()

        figname += profile_list[list_prop[1][i]] + '_' + list_prop[0][i] + '_' + str(list_prop[6][i])+\
            '_Truncate_' + str(list_prop[2][i]) + '_Cparam_' + str(list_prop[3][i]) + '_bmin_' +\
            str(list_prop[4][i]) + '_Nobs_' + str(list_prop[5][i]) + '_'

        labels.append(profile_list[list_prop[1][i]][0] + ' ' + tlab + ' ' +
                      list_prop[0][i] + ' ' + str(list_prop[6][i])[0] + ' C'
                      + str(list_prop[3][i]) + ' bmin' +
                      str(list_prop[4][i]) )

    fig = plt.figure()
    ax = plt.gca()
    for i in range(max_length):
        plt.plot(mt[:,i], lim[:,i], lw=2, color=color_list[i], label=labels[i])
    figname += 'CL_' + str(CL) + '.pdf'
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.legend()
    pl.xlim([10., 2.0 * 1000.])
    pl.ylim([10 ** -27., 10 ** -23.])
    pl.ylabel(r'$<$ $\sigma$ v $>$   [$cm^3/s$] ', fontsize=fs)
    pl.xlabel(r'$m_{\chi}$  [$GeV$]', fontsize=fs)
    fig.set_tight_layout(True)
    if save_fig:
        pl.savefig(figname)
    return

class LimitPlotter(object):

    def __init__(self, annih_prod='BB', n_obs=0., CL=0.95, pointlike=True,
                 alpha=0.16, profile=0, truncate=True, arxiv_num=13131729, b_min=20.,
                 mass_low=1., mass_high=3., fs=20, save_plot=True):

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
        self.save_plot = save_plot

        if self.pointlike:
            plike_tag = '_Pointlike'
        else:
            plike_tag = '_Extended'

        self.file_name = 'Limits' + plike_tag + '_' + self.profile_name + '_Truncate_' + \
                         str(self.truncate) + '_alpha_' + str(self.alpha) + \
                         '_annih_prod_' + self.annih_prod + '_arxiv_num_' + \
                         str(self.arxiv_num) + '_bmin_' + str(self.b_min) + '.dat'

    def check_if_lim_file_exists(self):
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
        #  lim_interp = interp1d(lim_file[:, 0], lim_file[:, 1], kind='cubic', bounds_error=False)
        mass_tab = np.logspace(self.mass_low, self.mass_high, 300)
        # limpt = lim_interp(mass_tab)
        limpt = interpola(mass_tab, lim_file[:, 0], lim_file[:, 1])

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

        plt.text(15., 4. * 10 ** -24., '{}\n{}\n{}'.format(plabel, trunclabel, label),
                 fontsize=15, ha='left', va='center')

        pl.xlim([10., 3.0 * 1000.])
        pl.ylim([10 ** -27., 10 ** -23.])
        pl.ylabel(r'$<$ $\sigma$ v $>$   [$cm^3/s$] ', fontsize=self.fs)
        pl.xlabel(r'$m_{\chi}$  [$GeV$]', fontsize=self.fs)
        figname = self.folder + '../Plots/' + 'Subhalo_Limits_{}_{}_'.format(self.profile_name, self.truncate) + \
            '{}_AnnihProd_{}_Cparams_{}_'.format(self.pointlike, self.annih_prod, self.arxiv_num) + \
            'bmin_{}_CL_{}_Nobs_{}.pdf'.format(self.b_min, self.CL, self.nobs)

        fig.set_tight_layout(True)
        if self.save_plot:
            pl.savefig(figname)
#        return np.column_stack((mass_tab, limpt))
        return mass_tab, limpt


class model_plots(object):
    def __init__(self, cross_sec=3. * 10 ** -26., mx=100., annih_prod='BB', pointlike=True, alpha=0.16, profile=0,
                 truncate=True, arxiv_num=10070438, mass_low=-5., mass_high=7., fs=20, b_min = 20.):
        profile_list = ['Einasto', 'NFW']
        self.cross_sec = cross_sec
        self.mx = mx
        self.annih_prod = annih_prod
        self.pointlike = pointlike
        self.alpha = alpha
        self.profile = profile
        self.profile_name = profile_list[profile]
        self.truncate = truncate
        self.arxiv_num = arxiv_num
        self.mass_low = mass_low
        self.mass_high = mass_high
        self.fs = fs
        self.b_min = b_min
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"

    def constant_flux_contours(self, threshold=10 ** -9., flux_low=-18., flux_high=6.):

        fluxtab = np.power(10., np.linspace(flux_low, flux_high, int(flux_high - flux_low) + 1))
        mass_tab = np.logspace(self.mass_low, self.mass_high, 50)
        mass_dist_tab = np.zeros(mass_tab.size * fluxtab.size).reshape((mass_tab.size, fluxtab.size))
        real_dmax = np.zeros(mass_tab.size)
        extend_dmax = np.zeros(mass_tab.size)
        for i, m in enumerate(mass_tab):
            if self.truncate:
                mm = m / 0.005
            else:
                mm = m
            model = Model(self.mx, self.cross_sec, self.annih_prod, mm, self.alpha,
                          truncate=self.truncate, arxiv_num=self.arxiv_num,
                          profile=self.profile, pointlike=True)
            real_dmax[i] = model.d_max_point(threshold=threshold)

            #  model2 = Model(self.mx, self.cross_sec, self.annih_prod, mm, self.alpha,
            #                 truncate=self.truncate, arxiv_num=self.arxiv_num,
            #                 profile=self.profile, pointlike=False)
            #  extend_dmax[i] = model2.D_max_extend()

            for j, f in enumerate(fluxtab):
                mass_dist_tab[i, j] = model.d_max_point(threshold=f)

        return real_dmax, extend_dmax, mass_dist_tab

    def plot_flux_contours(self, threshold=10 ** -9., flux_low=-18., flux_high=6.):
        # TODO: Find way to determine spatial extension boundary
        dmax, dmax_ext, contours = self.constant_flux_contours(threshold=threshold,
                                                               flux_low=flux_low,
                                                               flux_high=flux_high)

        mass_tab = np.logspace(self.mass_low, self.mass_high, 50)
        mlen, flen = np.shape(contours)

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()

        plt.plot(mass_tab, dmax, lw=2, color='Black')
        plt.plot(mass_tab, dmax_ext, lw=2, color='Black')
        for i in range(flen):
            plt.plot(mass_tab, contours[:, i], lw=1, color='Black', alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10 ** self.mass_low, 10 ** self.mass_high])
        pl.ylim([10 ** -4., 10.])
        Profile_List = ["Einasto", "NFW"]
        profile_name = Profile_List[self.profile]
        plt.title('{}'.format(profile_name), fontsize=20)
        if self.pointlike:
            plabel = 'Pointlike'
        else:
            plabel = 'Extended'
        if self.truncate:
            trunclabel = 'Truncated'
        else:
            trunclabel = 'SchoonenbergEtal'

        plt.text(2. * 10 ** -5, 1., '{}\n{}\n'.format(plabel, trunclabel) + r'$m_\chi = ${}'.format(self.mx) +
                                    '\n' + r'$<\sigma v> = $' + '{} \n '.format(self.cross_sec) +
                                    '$F_\gamma = ${}'.format(threshold),
                                    fontsize=15, ha='left', va='center')

        pl.ylabel('Distance  [kpc]', fontsize=self.fs)
        pl.xlabel(r'$M_{subhalo}$  [$M_\odot$]', fontsize=self.fs)

        figname = self.folder + '../Plots/' + 'DistanceCountours_{}_{}_'.format(profile_name, trunclabel) + \
            '{}_AnnihProd_{}_CrossSection_{}'.format(plabel, self.annih_prod, self.cross_sec) + \
            '_Cparams_{}_.pdf'.format(self.arxiv_num)

        #  adjustFigAspect(fig,aspect=.5)
        fig.set_tight_layout(True)
        pl.savefig(figname)
        return

    def dmax_obs_splice(self):

        Profile_names = ['Einasto', 'NFW']
        if self.pointlike:
            ptag = '_Pointlike'
        else:
            ptag = '_Extended'
        info_str = "Observable_Profile_" + str(Profile_names[self.profile]) + "_Truncate_" + \
                   str(self.truncate) + ptag + "_mx_" + str(self.mx) + "_annih_prod_" + \
                   self.annih_prod + "_arxiv_num_" + str(self.arxiv_num) + "/"

        folder = self.folder + info_str
        openfile = open(folder + "param_list.pkl", 'rb')
        dict_info = pickle.load(openfile)
        openfile.close()
        if self.truncate:
            mlow = np.log10(0.005) + dict_info['m_low']
            mhigh = np.log10(0.005) + dict_info['m_high']
            mass_list = np.logspace(mlow, mhigh, dict_info['m_num'])
        else:
            mass_list = np.logspace(dict_info['m_low'], dict_info['m_high'], dict_info['m_num'])
        c_list = np.logspace(dict_info['c_low'], dict_info['c_high'], dict_info['c_num'])
        if self.pointlike:
            file_name = 'Dmax_POINTLIKE_' + str(Profile_names[self.profile]) + '_Truncate_' + \
                        str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' + \
                        str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' + \
                        str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'
        else:
            file_name = 'Dmax_Extended' + str(Profile_names[self.profile]) + '_Truncate_' + \
                        str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' + \
                        str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' + \
                        str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'

        integrand_table = np.loadtxt(folder + file_name)
        m_num = mass_list.size
        c_num = c_list.size
        int_prep_spline = np.reshape(integrand_table[:, 2], (m_num, c_num))
        dmax = RectBivariateSpline(mass_list, c_list, int_prep_spline)
        fixinterp = interp2d(integrand_table[:, 0], integrand_table[:, 1], integrand_table[:, 2])
        f, ax = plt.subplots(5, 2, sharex='col', sharey='row')

        m = np.logspace(np.log10(mass_list[0]), np.log10(mass_list[-1]), 200)
        points = np.zeros(c_num * m.size / 2).reshape((c_num / 2, m.size))
        j = 0

        for i in range(c_num / 2):
            if i == 5:
                j = 1
            ii = i % 5
            points[i] = dmax.ev(m, c_list[2 * i])
            ax[ii, j].plot(m, points[i], 'b--', mass_list, fixinterp(mass_list, c_list[2 * i]), 'r.', ms=3)
            ax[ii, j].set_xscale("log")
            ax[ii, j].set_yscale('log')

        pl.xlim(3.24 * 10 ** 4., 10. ** 7.)
        f.set_tight_layout(True)
        pl.savefig('/Users/SamWitte/Desktop/example.pdf')
        return

    def c_sec_v_nobs_plot(self):
        # TODO Make multi plot possible
        if self.pointlike:
            ptag = '_Pointlike'
        else:
            ptag = '_Extended'

        dir = self.folder + '/Cross_v_Nobs/'
        file_name = self.profile_name + '_Truncate_' + str(self.truncate) + '_Cparam_' +\
            str(self.arxiv_num) + '_alpha_' + str(self.alpha) + '_mx_' + str(self.mx) +\
            '_annih_prod_' + self.annih_prod + '_bmin_' + str(self.b_min) + ptag +\
            '.dat' \

        try:
            c_vs_n = np.loadtxt(dir + file_name)
        except:
            c_vs_n = []
            print file_name
            print 'File Does Not Exist.'
            exit()
        c_list = np.logspace(np.log10(c_vs_n[0, 0]), np.log10(c_vs_n[-1, 0]), 200)
        nobs_list = interpola(c_list, c_vs_n[:, 0], c_vs_n[:, 1])

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10. ** -27., 2. * 10 ** -24.])
        pl.ylim([10 ** -1., 10. ** 2.])
        plt.plot(c_list, nobs_list, lw=2, color='Black')

        pl.xlabel(r'$<\sigma v> $  [$cm^3/s$]', fontsize=self.fs)
        pl.ylabel(r'$N_{obs}$', fontsize=self.fs)

        figname = self.folder + '../Plots/' + 'CrossSec_vs_Nobs_' + self.profile_name +\
            '_Truncate_' + str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
            str(self.alpha) + '_mx_' + str(self.mx) + '_annih_prod_' + self.annih_prod +\
            '_bmin_' + str(self.b_min) + ptag + '.pdf' \

        fig.set_tight_layout(True)
        pl.savefig(figname)
        return c_list, nobs_list


def plot_spectrum(mx=100., annih_prod='BB'):
    file_path = MAIN_PATH + "/Spectrum/"
    file_path += '{}'.format(int(mx)) + 'GeV_' + annih_prod + '_DMspectrum.dat'

    spectrum = np.loadtxt(file_path)
    e_gamma_tab = np.logspace(spectrum[0, 0], spectrum[-1, 0], 300)
    max_lim = np.max(spectrum[:, 0] ** 2. * spectrum[:, 1] / 10 ** .5)
    spec_plot = interpola(e_gamma_tab, spectrum[:, 0], spectrum[:, 0] ** 2. * spectrum[:, 1] / (10**.5 * max_lim))

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10. ** -1., 10. ** 2.])
    pl.ylim([10. ** -1.5, 1.2])
    plt.plot(e_gamma_tab, spec_plot, lw=2, color='Black')

    pl.xlabel(r'$E_\gamma$  [GeV]', fontsize=20)
    pl.ylabel(r'$E_\gamma^2 dN / dE_\gamma$  [GeV $cm^2$ / s]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    figname = folder + '../Plots/' + 'GammaSpectrum_mx_' + str(mx) + '_annih_prod_' +\
        annih_prod + '.pdf'
    fig.set_tight_layout(True)
    pl.savefig(figname)
    return
