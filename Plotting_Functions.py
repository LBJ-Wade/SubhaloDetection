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
from scipy.optimize import fminbound
import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
from profiles import Einasto, NFW

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

        fig_name = self.folder + '../Plots/' + 'CrossSec_vs_Nobs_' + self.profile_name +\
            '_Truncate_' + str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
            str(self.alpha) + '_mx_' + str(self.mx) + '_annih_prod_' + self.annih_prod +\
            '_bmin_' + str(self.b_min) + ptag + '.pdf'

        fig.set_tight_layout(True)
        pl.savefig(fig_name)
        return c_list, nobs_list


def plot_spectrum(mx=100., annih_prod='BB'):
    file_path = MAIN_PATH + "/Spectrum/"
    file_path += '{}'.format(int(mx)) + 'GeV_' + annih_prod + '_DMspectrum.dat'

    spectrum = np.loadtxt(file_path)
    imax = 0
    for i in range(len(spectrum)):
        if spectrum[i, 1] < 10 or i == (len(spectrum) - 1):
            imax = i
            break

    spectrum = spectrum[0:imax, :]
    Nevents = 10. ** 5.
    spectrum[:, 1] /= Nevents
    e_gamma_tab = np.logspace(np.log10(spectrum[0, 0]), np.log10(spectrum[-1, 0]), 200)
    max_lim = np.max(np.array([spectrum[:, 0] ** 2. * spectrum[:, 1]]))
    spec_plot = interpola(e_gamma_tab, spectrum[:, 0], spectrum[:, 0] ** 2. * spectrum[:, 1] / max_lim)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10. ** -1., 10. ** 2.])
    pl.ylim([10. ** -1., 1.2])
    plt.plot(e_gamma_tab, spec_plot, '--', lw=2, color='Black')

    pl.xlabel(r'$E_\gamma$  [GeV]', fontsize=20)
    pl.ylabel(r'$E_\gamma^2 dN / dE_\gamma$  [GeV $cm^2$ / s]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'GammaSpectrum_mx_' + str(mx) + '_annih_prod_' +\
        annih_prod + '.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def plot_profiles(m_sub=10.**7., arxiv_num=[13131729], density_sqr=False, M200=[False]):

    if len(arxiv_num) == 1:
        arxiv_num *= 4
    elif len(arxiv_num) == 2:
        arxiv_num *= 2
    elif len(arxiv_num) == 3:
        print 'I dont know what to do with arxiv num input...'
        exit()
    if len(M200) == 1:
        M200 *= 4
    elif len(M200) == 2:
        M200 *= 2
    elif len(M200) == 3:
        print 'I dont know what to do with M200 num input...'
        exit()

    minrad = -2.
    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=arxiv_num[0], M200=M200[0])
    n_tr = NFW(m_sub / 0.005, .16, truncate=True, arxiv_num=arxiv_num[1], M200=M200[1])
    e_ntr = Einasto(m_sub, .16, truncate=False, arxiv_num=arxiv_num[2], M200=M200[2])
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=arxiv_num[3], M200=M200[3])

    r1_e_tr = np.logspace(minrad, np.log10(e_tr.max_radius), 100)
    r2_e_tr = np.logspace(np.log10(e_tr.max_radius), np.log10(e_tr.virial_radius), 100)
    r1_n_tr = np.logspace(minrad, np.log10(n_tr.max_radius), 100)
    r2_n_tr = np.logspace(np.log10(n_tr.max_radius), np.log10(n_tr.virial_radius), 100)
    r_e_ntr = np.logspace(minrad, np.log10(e_ntr.max_radius), 200)
    r2_e_ntr = np.logspace(np.log10(e_tr.max_radius), np.log10(e_ntr.virial_radius), 100)
    r_n_ntr = np.logspace(minrad, np.log10(n_ntr.max_radius), 200)
    r2_n_ntr = np.logspace(np.log10(e_tr.max_radius), np.log10(n_ntr.virial_radius), 100)

    if not density_sqr:
        den_e_tr1 = e_tr.density(r1_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_e_tr2 = e_tr.density(r2_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_n_tr1 = n_tr.density(r1_n_tr) * GeVtoSolarM * kpctocm ** 3.
        den_n_tr2 = n_tr.density(r2_n_tr) * GeVtoSolarM * kpctocm ** 3.
        den_e_ntr = e_ntr.density(r_e_ntr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr = n_ntr.density(r_n_ntr) * GeVtoSolarM * kpctocm ** 3.
        den_e_ntr2 = e_ntr.density(r2_e_ntr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr2 = n_ntr.density(r2_n_ntr) * GeVtoSolarM * kpctocm ** 3.
    else:
        den_e_tr1 = e_tr.density(r1_e_tr) ** 2.
        den_e_tr2 = e_tr.density(r2_e_tr) ** 2.
        den_n_tr1 = n_tr.density(r1_n_tr) ** 2.
        den_n_tr2 = n_tr.density(r2_n_tr) ** 2.
        den_e_ntr = e_ntr.density(r_e_ntr) ** 2.
        den_n_ntr = n_ntr.density(r_n_ntr) ** 2.
        den_e_ntr2 = e_ntr.density(r2_e_ntr) ** 2.
        den_n_ntr2 = n_ntr.density(r2_n_ntr) ** 2.

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** minrad, 3. * 10. ** 1.])
    plt.plot(r1_e_tr, den_e_tr1, lw=1, color='Black', label='Einasto, T')
    plt.plot(r2_e_tr, den_e_tr2, '.', ms=1, color='Black')
    plt.plot(r1_n_tr, den_n_tr1, lw=1, color='Blue', label='NFW, T')
    plt.plot(r2_n_tr, den_n_tr2, '.', ms=1, color='Blue')
    plt.plot(r_e_ntr, den_e_ntr, lw=1, color='Red', label='Einasto, NT')
    plt.plot(r2_e_ntr, den_e_ntr2, '.', ms=1, lw=1, color='Red')
    plt.plot(r_n_ntr, den_n_ntr, lw=1, color='Magenta', label='NFW, NT')
    plt.plot(r2_n_ntr, den_n_ntr2, '.', ms=1, lw=1, color='Magenta')
    plt.legend()

    pl.xlabel('Radius [kpc]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    if not density_sqr:
        pl.ylabel(r'$\rho [M_\odot / kpc^3$]', fontsize=20)
        pl.ylim([10. ** 3., 10. ** 11.])
        fig_name = folder + '../Plots/' + 'Density_Comparison_msubhalo_' + str(m_sub) + \
            '_arxiv_nums_' + str(arxiv_num[0]) + '_' + str(arxiv_num[1]) + '_' + \
            str(arxiv_num[2]) + '_' + str(arxiv_num[3]) + '.pdf'
    else:
        pl.ylabel(r'$\rho^2$ [$GeV^2 / cm^6$]', fontsize=20)
        pl.ylim([10. ** -8., 10. ** 6.])
        fig_name = folder + '../Plots/' + 'DensitySqr_Comparison_msubhalo_' + str(m_sub) + \
            '_arxiv_nums_' + str(arxiv_num[0]) + '_' + str(arxiv_num[1]) + '_' + \
            str(arxiv_num[2]) + '_' + str(arxiv_num[3]) + '.pdf'

    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def Jfactor_plots(m_sub=10.**7., arxiv_num=[13131729], M200=[False], mx=100.,
                  cross_sec=3.*10**-26., annih_prod='BB'):

    if len(arxiv_num) == 1:
        arxiv_num *= 4
    elif len(arxiv_num) == 2:
        arxiv_num *= 2
    elif len(arxiv_num) == 3:
        print 'I dont know what to do with arxiv num input...'
        exit()
    if len(M200) == 1:
        M200 *= 4
    elif len(M200) == 2:
        M200 *= 2
    elif len(M200) == 3:
        print 'I dont know what to do with M200 num input...'
        exit()

    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=arxiv_num[0], M200=M200[0])
    mod_e_tr = Model(mx, cross_sec, annih_prod, m_sub / 0.005, 0.16,
                     truncate=True, arxiv_num=arxiv_num[0], profile=0, m200=M200[0])
    n_tr = NFW(m_sub / 0.005, .16, truncate=True, arxiv_num=arxiv_num[1], M200=M200[1])
    mod_n_tr = Model(mx, cross_sec, annih_prod, m_sub / 0.005, 0.16,
                     truncate=True, arxiv_num=arxiv_num[1], profile=1, m200=M200[0])
    e_ntr = Einasto(m_sub, .16, truncate=False, arxiv_num=arxiv_num[2], M200=M200[2])
    mod_e_ntr = Model(mx, cross_sec, annih_prod, m_sub, 0.16,
                      truncate=False, arxiv_num=arxiv_num[2], profile=0, m200=M200[0])
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=arxiv_num[3], M200=M200[3])
    mod_n_ntr = Model(mx, cross_sec, annih_prod, m_sub, 0.16,
                      truncate=False, arxiv_num=arxiv_num[3], profile=1, m200=M200[0])

    def se_boundary(dist, prof, min_ext=0.1):
        return np.abs(prof.Spatial_Extension(10. ** dist) - min_ext)

    se_bound_e_tr = mod_e_tr.D_max_extend()
    print 'Spatial Extension Boundary Einasto (Truncated): ', se_bound_e_tr
    se_bound_n_tr = mod_n_tr.D_max_extend()
    print 'Spatial Extension Boundary NFW (Truncated): ', se_bound_n_tr
    se_bound_e_ntr = mod_e_ntr.D_max_extend()
    print 'Spatial Extension Boundary Einasto (Not Truncated): ', se_bound_e_ntr
    se_bound_n_ntr = mod_n_ntr.D_max_extend()
    print 'Spatial Extension Boundary NFW (Not Truncated): ', se_bound_n_ntr

    num_dist_pts = 30
    dist_tab = np.logspace(-3., 1., num_dist_pts)
    e_tr_j = np.zeros(dist_tab.size * 2).reshape((dist_tab.size, 2))
    n_tr_j = np.zeros(dist_tab.size * 2).reshape((dist_tab.size, 2))
    e_ntr_j = np.zeros(dist_tab.size * 2).reshape((dist_tab.size, 2))
    n_ntr_j = np.zeros(dist_tab.size * 2).reshape((dist_tab.size, 2))
    for i, d in enumerate(dist_tab):
        print i+1, '/', num_dist_pts
        if d <= se_bound_e_tr:
            e_tr_j[i] = [d, np.power(10, e_tr.J_pointlike(d))]
        if d <= se_bound_n_tr:
            n_tr_j[i] = [d, np.power(10, n_tr.J_pointlike(d))]
        if d <= se_bound_e_ntr:
            e_ntr_j[i] = [d, np.power(10, e_ntr.J_pointlike(d))]
        if d <= se_bound_n_ntr:
            n_ntr_j[i] = [d, np.power(10, n_ntr.J_pointlike(d))]

    e_tr_j = e_tr_j[~np.all([e_tr_j[:, 0] == 0], axis=0)]
    n_tr_j = n_tr_j[~np.all([n_tr_j[:, 0] == 0], axis=0)]
    e_ntr_j = e_ntr_j[~np.all([e_ntr_j[:, 0] == 0], axis=0)]
    n_ntr_j = n_ntr_j[~np.all([n_ntr_j[:, 0] == 0], axis=0)]

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    pl.xlim([10 ** -4., 10. ** 1.])
    pl.ylim([10 ** 20., np.max([e_tr_j[0], n_tr_j[0]])])
    plt.plot(e_tr_j[:, 0], e_tr_j[:, 1], lw=1, color='Black', label='Einasto, T')
    plt.plot(n_tr_j[:, 0], n_tr_j[:, 1], lw=1, color='Blue', label='NFW, T')
    plt.plot(e_ntr_j[:, 0], e_ntr_j[:, 1], lw=1, color='Red', label='Einasto, NT')
    plt.plot(n_ntr_j[:, 0], n_ntr_j[:, 1], lw=1, color='Magenta', label='NFW, NT')

    plt.axvline(x=se_bound_e_tr, ymin=0., ymax=1., linewidth=1, color='Black', alpha=0.2)
    plt.axvline(x=se_bound_n_tr,  ymin=0., ymax=1., linewidth=1, color='Blue', alpha=0.2)
    plt.axvline(x=se_bound_e_ntr,  ymin=0., ymax=1., linewidth=1, color='Red', alpha=0.2)
    plt.axvline(x=se_bound_n_ntr,  ymin=0., ymax=1., linewidth=1, color='Magenta', alpha=0.2)

    plt.legend()

    pl.xlabel('Distance [kpc]', fontsize=20)
    pl.ylabel(r'J   [$GeV^2 / cm^5$]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Jfactor_Comparison_msubhalo_' + str(m_sub) + \
               '_arxiv_nums_' + str(arxiv_num[0]) + '_' + str(arxiv_num[1]) + '_' + \
               str(arxiv_num[2]) + '_' + str(arxiv_num[3]) + '.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def extension_vs_dist(m_sub=1.*10**7., arxiv_num=[13131729], M200=[False]):

    #  NFW nontruncate having issues
    if len(arxiv_num) == 1:
        arxiv_num *= 4
    elif len(arxiv_num) == 2:
        arxiv_num *= 2
    elif len(arxiv_num) == 3:
        print 'I dont know what to do with arxiv num input...'
        exit()
    if len(M200) == 1:
        M200 *= 4
    elif len(M200) == 2:
        M200 *= 2
    elif len(M200) == 3:
        print 'I dont know what to do with arxiv num input...'
        exit()

    mindist = -1.5
    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=arxiv_num[0], M200=M200[0])
    n_tr = NFW(m_sub / 0.005, .16, truncate=True, arxiv_num=arxiv_num[1], M200=M200[1])
    e_ntr = Einasto(m_sub, .16, truncate=False, arxiv_num=arxiv_num[2], M200=M200[2])
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=arxiv_num[3], M200=M200[3])

    num_dist_pts = 30
    dist_tab = np.logspace(mindist, 1.5, num_dist_pts)

    e_tr_se = np.zeros(dist_tab.size)
    n_tr_se = np.zeros(dist_tab.size)
    e_ntr_se = np.zeros(dist_tab.size)
    n_ntr_se = np.zeros(dist_tab.size)
    for i,d in enumerate(dist_tab):
        print i+1, '/', num_dist_pts
        e_tr_se[i] = e_tr.Spatial_Extension(d)
        n_tr_se[i] = n_tr.Spatial_Extension(d)
        e_ntr_se[i] = e_ntr.Spatial_Extension(d)
        n_ntr_se[i] = n_ntr.Spatial_Extension(d)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    print 'Einasto Truncated: ', np.column_stack((dist_tab, e_tr_se))
    print 'NFW Truncated: ', np.column_stack((dist_tab, n_tr_se))
    print 'Einasto Not Truncated: ', np.column_stack((dist_tab, e_ntr_se))
    print 'NFW Not Truncated: ', np.column_stack((dist_tab, n_ntr_se))

    dist_plot = np.logspace(mindist, 1., 200)
    e_tr_plot = interpola(dist_plot, dist_tab, e_tr_se)
    n_tr_plot = interpola(dist_plot, dist_tab, n_tr_se)
    e_ntr_plot = interpola(dist_plot, dist_tab, e_ntr_se)
    n_ntr_plot = interpola(dist_plot, dist_tab, n_ntr_se)

    def rad_ext(dist, prof):
        return radtodeg * np.arctan(prof.max_radius / dist)
    rad_ex_e_tr = rad_ext(dist_plot, e_tr)
    rad_ex_n_tr = rad_ext(dist_plot, n_tr)
    rad_ex_e_ntr = rad_ext(dist_plot, e_ntr)
    rad_ex_n_ntr = rad_ext(dist_plot, n_ntr)

    pl.xlim([10 ** mindist, 8. * 10. ** 1.])
    pl.ylim([np.min([e_tr_se[-1], n_tr_se[-1], e_ntr_se[-1], n_ntr_se[-1]]),
             3.0 * np.max([e_tr_se[0], n_tr_se[0], e_ntr_se[0], n_ntr_se[0]])])
    plt.plot(dist_plot, e_tr_plot, lw=1, color='Black', label='Einasto, T')
    plt.plot(dist_plot, rad_ex_e_tr, '--', ms=1, color='Black')
    plt.plot(dist_plot, n_tr_plot, lw=1, color='Blue', label='NFW, T')
    plt.plot(dist_plot, rad_ex_n_tr, '--', ms=1, color='Blue')
    plt.plot(dist_plot, e_ntr_plot, lw=1, color='Red', label='Einasto, NT')
    plt.plot(dist_plot, rad_ex_e_ntr, '--', ms=1, color='Red')
    plt.plot(dist_plot, n_ntr_plot, lw=1, color='Magenta', label='NFW, NT')
    plt.plot(dist_plot, rad_ex_n_ntr, '--', ms=1, color='Magenta')

    plt.text(9., 4. * 10 ** -25., r'$\theta = 0.1^\circ$', fontsize=15, ha='right', va='center')
    plt.legend()

    pl.xlabel('Distance [kpc]', fontsize=20)
    pl.ylabel('Spatial Extension [degrees]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'SpatialExt_Distance_msubhalo_' + str(m_sub) + \
        '_arxiv_nums_' + str(arxiv_num[0]) + '_' + str(arxiv_num[1]) + '_' + \
        str(arxiv_num[2]) + '_' + str(arxiv_num[3]) + '.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return

