# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from subhalo import *
import os
import pickle
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fminbound
import matplotlib as mpl
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
from profiles import *

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

        figname += Profile_list[list_prop[1][i]] + '_' + list_prop[0][i] + '_' + str(list_prop[6][i])+\
            '_Truncate_' + str(list_prop[2][i]) + '_Cparam_' + str(list_prop[3][i]) + '_bmin_' +\
            str(list_prop[4][i]) + '_Nobs_' + str(list_prop[5][i]) + '_'

        labels.append(Profile_list[list_prop[1][i]][0] + ' ' + tlab + ' ' +
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

        self.profile_name = Profile_list[profile]
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
                 truncate=True, arxiv_num=10070438, mass_low=-5., mass_high=7., fs=20, b_min=20.,
                 gam=0.85, stiff_rb=False, m_low=10**4):

        self.cross_sec = cross_sec
        self.mx = mx
        self.annih_prod = annih_prod
        self.pointlike = pointlike
        self.alpha = alpha
        self.profile = profile
        self.profile_name = Profile_list[profile]
        self.truncate = truncate
        self.arxiv_num = arxiv_num
        self.mass_low = mass_low
        self.mass_high = mass_high
        self.fs = fs
        self.b_min = b_min
        self.gam = gam
        self.stiff_rb = stiff_rb
        self.m_low = np.log10(m_low)
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
        profile_name = Profile_list[self.profile]
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

        if self.pointlike:
            ptag = '_Pointlike'
        else:
            ptag = '_Extended'
        if self.profile == 2:
            self.extra_tag = '_gamma_{:.2f}_Conser_'.format(self.gam) + '_stiff_rb_' + str(self.stiff_rb)
        else:
            self.extra_tag = '_arxiv_' + str(self.arxiv_num)
        if self.truncate:
            self.tr_tag = '_Truncate_'
        else:
            self.tr_tag = ''

        info_str = "Observable_Profile_" + self.profile_name + self.tr_tag +\
            ptag + "_mx_" + str(self.mx) + "_annih_prod_" +\
            self.annih_prod + self.extra_tag + '_Mlow_{:.3e}'.format(self.m_low) + "/"
        folder = self.folder + info_str

        file_name = 'Dmax_POINTLIKE_HW_mx_200.0_cross_sec_1.000e-27_annih_prod_BB_gamma_0.85_Conser__stiff_rb_False.dat'

        integrand_table = np.loadtxt(folder + file_name)
        mass_list = np.unique(integrand_table[:, 0])
        c_list = np.unique(integrand_table[:, 1])
        m_num = mass_list.size
        c_num = c_list.size
        if self.profile == 0:
            int_prep_spline = np.reshape(integrand_table[:, 2], (m_num, c_num))
            dmax = RectBivariateSpline(mass_list, c_list, int_prep_spline)
            fixinterp = interp2d(integrand_table[:, 0], integrand_table[:, 1], integrand_table[:, 2])
        else:
            dmax = UnivariateSpline(mass_list, integrand_table[:, 1])
            fixinterp = interp1d(integrand_table[:,0], integrand_table[:, 1])
        f, ax = plt.subplots(5, 2, sharex='col', sharey='row')

        m = np.logspace(np.log10(mass_list[0]), np.log10(mass_list[-1]), 200)
        points = np.zeros(c_num * m.size / 2).reshape((c_num / 2, m.size))
        j = 0

        for i in range(c_num / 2):
            if i == 5:
                j = 1
            ii = i % 5
            #points[i] = dmax.ev(m, c_list[2 * i])
            points[i] = dmax(m)
            #ax[ii, j].plot(m, points[i], 'b--', mass_list, fixinterp(mass_list, c_list[2 * i]), 'r.', ms=3)
            ax[ii, j].plot(m, points[i], 'b--', mass_list, fixinterp(mass_list), 'r.', ms=3)
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
        mlow = 3.24 * 10**4.
        dir = self.folder + '/Cross_v_Nobs/'
        file_name = self.profile_name + '_mx_' + str(self.mx) +\
            '_annih_prod_' + self.annih_prod + '_bmin_' + str(self.b_min) + ptag +\
            '_Truncate_' + str(self.truncate) + '_Cparam_' +\
            str(self.arxiv_num) + '_alpha_' + str(self.alpha) + '_Mlow_{:.3f}'.format(np.log10(mlow)) + '.dat'

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


def plot_profiles(m_sub=10.**7., density_sqr=False):

    minrad = -2.
    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True)
    hw_fit = HW_Fit(m_sub, M200=True, gcd=8.5)

    hw_fit_gl = HW_Fit(m_sub, gam=0.426, M200=True, gcd=8.5)
    hw_fit_gh = HW_Fit(m_sub, gam=1.316, M200=True, gcd=8.5)

    r1_e_tr = np.logspace(minrad, np.log10(e_tr.max_radius), 100)
    r2_e_tr = np.logspace(np.log10(e_tr.max_radius), np.log10(e_tr.virial_radius), 100)
    r_n_ntr = np.logspace(minrad, np.log10(n_ntr.max_radius), 100)
    r2_n_ntr = np.logspace(np.log10(n_ntr.max_radius), np.log10(n_ntr.virial_radius), 20)
    r_hw = np.logspace(minrad, np.log10(hw_fit.max_radius), 100)
    r_hw_gl = np.logspace(minrad, np.log10(hw_fit_gl.max_radius), 100)
    r_hw_gh = np.logspace(minrad, np.log10(hw_fit_gh.max_radius), 100)

    if not density_sqr:
        den_e_tr1 = e_tr.density(r1_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_e_tr2 = e_tr.density(r2_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr = n_ntr.density(r_n_ntr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr2 = n_ntr.density(r2_n_ntr) * GeVtoSolarM * kpctocm ** 3.
        den_hw = hw_fit.density(r_hw) * GeVtoSolarM * kpctocm ** 3.
        den_hw_gl = hw_fit_gl.density(r_hw_gl) * GeVtoSolarM * kpctocm ** 3.
        den_hw_gh = hw_fit_gh.density(r_hw_gh) * GeVtoSolarM * kpctocm ** 3.
    else:
        den_e_tr1 = e_tr.density(r1_e_tr) ** 2.
        den_e_tr2 = e_tr.density(r2_e_tr) ** 2.
        den_n_ntr = n_ntr.density(r_n_ntr) ** 2.
        den_n_ntr2 = n_ntr.density(r2_n_ntr) ** 2.
        den_hw = hw_fit.density(r_hw) ** 2.
        den_hw_gl = hw_fit.density(r_hw_gl) ** 2.
        den_hw_gh = hw_fit.density(r_hw_gh) ** 2.


    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** minrad, 3. * 10. ** 1.])
    plt.plot(r1_e_tr, den_e_tr1, lw=1, color='Black', label='Einasto, T')
    plt.plot(r2_e_tr, den_e_tr2, '.', ms=1, color='Black')
    plt.plot(r_n_ntr, den_n_ntr, lw=1, color='Magenta', label='NFW, NT')
    plt.plot(r2_n_ntr, den_n_ntr2, '.', ms=1, lw=1, color='Magenta')
    plt.plot(r_hw, den_hw, lw=1, color='Blue', label=r'$<HW>$')

    rb_max = np.zeros(len(r_hw_gh))
    rb_min = np.zeros(len(r_hw_gh))
    g_min = np.zeros(len(r_hw_gh))
    g_max = np.zeros(len(r_hw_gh))
    for i in range(r_hw_gh.size):
        g_min[i] = np.min([den_hw_gl[i], den_hw_gh[i]])
        g_max[i] = np.max([den_hw_gl[i], den_hw_gh[i]])
    plt.plot(r_hw_gh, den_hw_gl, '-.', r_hw_gh, den_hw_gh, '-.', color='k', alpha=0.5)
    ax.fill_between(r_hw_gh, g_min, g_max, where=g_max >= g_min, facecolor='Blue', interpolate=True, alpha=0.3)

    plt.legend()

    pl.xlabel('Radius [kpc]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    if not density_sqr:
        pl.ylabel(r'$\rho [M_\odot / kpc^3$]', fontsize=20)
        pl.ylim([10. ** 3., 10. ** 11.])
        fig_name = folder + '../Plots/' + 'Density_Comparison_msubhalo_{:.2e}'.format(m_sub) + \
            '_BLH_Bertone_HW.pdf'
    else:
        pl.ylabel(r'$\rho^2$ [$GeV^2 / cm^6$]', fontsize=20)
        pl.ylim([10. ** -8., 10. ** 6.])
        fig_name = folder + '../Plots/' + 'DensitySqr_Comparison_msubhalo_{:.2e}'.format(m_sub) + \
            '_BLH_Bertone_HW.pdf'

    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def Jfactor_plots(m_sub=10.**7., dist=1.):
    # J vs Dist
    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True)
    hw_fit = HW_Fit(m_sub, M200=True, cons=False, stiff_rb=False)


    num_dist_pts = 30
    dist_tab = np.logspace(-3., 1., num_dist_pts)
    e_tr_j = np.zeros(num_dist_pts* 2).reshape((num_dist_pts, 2))
    n_ntr_j = np.zeros(num_dist_pts * 2).reshape((num_dist_pts, 2))
    hw_j = np.zeros(num_dist_pts * 2).reshape((num_dist_pts, 2))
    hw_j_gl = np.zeros(num_dist_pts * 2).reshape((num_dist_pts, 2))
    hw_j_gh = np.zeros(num_dist_pts * 2).reshape((num_dist_pts, 2))
    n_sc_j = np.zeros(num_dist_pts)

    for i in range(num_dist_pts):
        print i+1, '/', num_dist_pts
        e_tr_j[i] = [dist_tab[i], np.power(10, e_tr.J_pointlike(dist_tab[i]))]
        n_ntr_j[i] = [dist_tab[i], np.power(10, n_ntr.J_pointlike(dist_tab[i]))]
        hw_j[i] = [dist_tab[i], np.power(10, hw_fit.J_pointlike(dist_tab[i]))]

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    pl.xlim([10 ** -3., 10. ** 1.])
    pl.ylim([10 ** 20., np.max([e_tr_j[0], hw_j[0]])])
    plt.plot(e_tr_j[:, 0], e_tr_j[:, 1], lw=1, color='Black', label='BLH')
    plt.plot(n_ntr_j[:, 0], n_ntr_j[:, 1], lw=1, color='Magenta', label='Bertone')
    plt.plot(hw_j[:, 0], hw_j[:, 1], lw=1, color='Blue', label='HW')


    #g_min = np.zeros(len(dist_tab))
    #g_max = np.zeros(len(dist_tab))
    #for i in range(dist_tab.size):
    #    g_min[i] = np.min([hw_j_gh[i, 1], hw_j_gl[i, 1]])
    #    g_max[i] = np.max([hw_j_gh[i, 1], hw_j_gl[i, 1]])
    #plt.plot(dist_tab, hw_j_gl[:, 1], '-.', dist_tab, hw_j_gh[:, 1], '-.', color='k')
    #ax.fill_between(dist_tab, g_min, g_max, where=g_max >= g_min, facecolor='Blue', interpolate=True, alpha=0.3)

    plt.legend()

    pl.xlabel('Distance [kpc]', fontsize=20)
    pl.ylabel(r'J   [$GeV^2 / cm^5$]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'J_vs_Distance_Msubhalo_{:.2e}'.format(m_sub) + \
        '_BLH_Bertone_HW.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)

    # J vs Mass
    mass_tab = np.logspace(0., 7., 150)
    e_tr_j = np.zeros(mass_tab.size)
    n_ntr_j = np.zeros(mass_tab.size)
    hw_j = np.zeros(mass_tab.size)
    hw_j_c = np.zeros(mass_tab.size)
    hw_j_o = np.zeros(mass_tab.size)
    hw_j_gl = np.zeros(mass_tab.size)
    hw_j_gh = np.zeros(mass_tab.size)
    sc_j = np.zeros(mass_tab.size)
    for i, m_sub in enumerate(mass_tab):
        print i + 1, '/', mass_tab.size
        e_tr_j[i] = 10. ** Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False).J_pointlike(dist)
        n_ntr_j[i] = 10. ** NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True).J_pointlike(dist)
        hw_j[i] = 10. ** HW_Fit(m_sub, M200=True, cons=False, stiff_rb=False).J_pointlike(dist)
        hw_j_gl[i] = 10. ** HW_Fit(m_sub, gam=0.426, M200=True).J_pointlike(dist)
        hw_j_gh[i] = 10. ** HW_Fit(m_sub, gam=1.316, M200=True).J_pointlike(dist)
        sc_j[i] = 10. ** NFW(m_sub, 1., truncate=False, arxiv_num=160304057, M200=True).J_pointlike(dist)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    pl.xlim([10 ** 0., 10. ** 7.])
    pl.ylim([np.min([hw_j_gl, n_ntr_j, hw_j_c]), np.max([e_tr_j, hw_j_o])])
    plt.plot(mass_tab, e_tr_j, lw=1, color='Black', label='BLH')
    plt.plot(mass_tab, n_ntr_j, lw=1, color='Magenta', label='Bertone')
    plt.plot(mass_tab, hw_j, lw=1, color='Blue', label='HW')
    plt.plot(mass_tab, sc_j, lw=1, color='Aqua', label='SC')

    rb_max = np.zeros(len(mass_tab))
    rb_min = np.zeros(len(mass_tab))
    g_min = np.zeros(len(mass_tab))
    g_max = np.zeros(len(mass_tab))
    for i in range(mass_tab.size):
        rb_min[i] = np.min([hw_j_c[i], hw_j_o[i]])
        rb_max[i] = np.max([hw_j_c[i], hw_j_o[i]])
        g_min[i] = np.min([hw_j_gh[i], hw_j_gl[i]])
        g_max[i] = np.max([hw_j_gh[i], hw_j_gl[i]])
    plt.plot(mass_tab, hw_j_gl, '-.', mass_tab, hw_j_gh, '-.', color='k')
    ax.fill_between(mass_tab, g_min, g_max, where=g_max >= g_min, facecolor='Blue', interpolate=True, alpha=0.3)

    pl.xlabel(r'Mass [$M_\odot$]', fontsize=20)
    pl.ylabel(r'J   [$GeV^2 / cm^5$]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Jfactor_vs_Mass_Distance_{:.2f}'.format(dist) + \
               '_BLH_Bertone_HW.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)

    return


def extension_vs_dist(m_sub=1.*10**7.):

    mindist = -1.
    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True)
    hw_fit = HW_Fit(m_sub, M200=True, gcd=8.5)
    hw_fit_gl = HW_Fit(m_sub, M200=True, gcd=8.5, gam=0.426)

    num_dist_pts = 15
    dist_tab = np.logspace(mindist, 1.5, num_dist_pts)

    e_tr_se = np.zeros(dist_tab.size)
    n_ntr_se = np.zeros(dist_tab.size)
    hw_se = np.zeros(dist_tab.size)
    hw_gl = np.zeros(dist_tab.size)
    for i,d in enumerate(dist_tab):
        print i+1, '/', num_dist_pts
        e_tr_se[i] = e_tr.Spatial_Extension(d)
        print d, e_tr_se[i]
        n_ntr_se[i] = n_ntr.Spatial_Extension(d)
        print d, n_ntr_se[i]
        hw_se[i] = hw_fit.Spatial_Extension(d)
        print d, hw_se[i]
        hw_gl[i] = hw_fit_gl.Spatial_Extension(d)
        print d, hw_gl[i]

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    dist_plot = np.logspace(mindist, 1., 200)
    e_tr_plot = interpola(dist_plot, dist_tab, e_tr_se)
    n_ntr_plot = interpola(dist_plot, dist_tab, n_ntr_se)
    hw_plot = interpola(dist_plot, dist_tab, hw_se)
    hw_gl_plot = interpola(dist_plot, dist_tab, hw_gl)

    def rad_ext(dist, prof):
        return radtodeg * np.arctan(prof.max_radius / dist)
    rad_ex_e_tr = rad_ext(dist_plot, e_tr)
    rad_ex_n_ntr = rad_ext(dist_plot, n_ntr)
    rad_ex_hw = rad_ext(dist_plot, hw_fit)
    rad_ex_hw_gl = rad_ext(dist_plot, hw_fit_gl)

    pl.xlim([10 ** mindist, 1. * 10. ** 1.])
    pl.ylim([10. ** -1., 90.])
    plt.plot(dist_plot, e_tr_plot, lw=1, color='Black', label='Einasto, T')
    plt.plot(dist_plot, rad_ex_e_tr, '--', ms=1, color='Black')
    plt.plot(dist_plot, n_ntr_plot, lw=1, color='Magenta', label='NFW, NT')
    plt.plot(dist_plot, rad_ex_n_ntr, '--', ms=1, color='Magenta')
    plt.plot(dist_plot, hw_plot, lw=1, color='Blue', label='HW')
    plt.plot(dist_plot, rad_ex_hw, '--', ms=1, color='Blue')

    plt.plot(dist_plot, hw_gl_plot, '-.', lw=1, color='Blue', alpha=0.5, label=r'HW $\gamma = 0.426$')
    plt.plot(dist_plot, rad_ex_hw_gl, '.', ms=1, color='Blue', alpha = 0.5)

    plt.legend()

    pl.xlabel('Distance [kpc]', fontsize=20)
    pl.ylabel('Spatial Extension [degrees]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'SpatialExt_Distance_msubhalo_{:.2e}'.format(m_sub) + \
        'BLH_Bertone_HW.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return

