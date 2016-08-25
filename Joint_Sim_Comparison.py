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
from scipy.optimize import curve_fit, minimize
from scipy.special import beta, kv

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
                    d_vmax_min=0., d_vmax_max=1., ms=3, plot=True):

        vl_ids = self.class_list[0].evolution_tracks(return_recent=True)
        color_list = ['Red', 'Blue']
        shape_list = ['o', 'd']

        collection = np.array([])
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
                if (i == 0 and (sub[0] in vl_ids)) or (i == 1 and (sub[-1] == 1)):
                    clr = 'k'
                    shape = shape_list[-1]
                    mss = 4
                    mew = 1
                else:
                    clr = color_list[i]
                    shape = shape_list[i]
                    mss=ms
                    mew = 0.5
                rvmax, vmax, mass = [sub[4], sub[3], sub[5]]
                try:
                    if self.names[i] == 'Via Lactea II':
                        err = np.sqrt(sub[5] * (4.1 * 10 ** 3.))
                    elif self.names[i] == 'ELVIS':
                        err = np.sqrt(sub[-3]) * sub[-2]
                    bf_gamma = find_gamma(mass, vmax, rvmax, error=err)
                except ValueError:
                    fail_count += 1
                    bf_gamma = 0.0
                subhalo_kmmdsm = KMMDSM(mass, bf_gamma, arxiv_num=160106781,
                                        vmax=vmax, rmax=rvmax)
                if i == 0 and j == 0:
                    collection = np.array([mass, bf_gamma, subhalo_kmmdsm.rb])
                    col_info = np.array([shape, mss, clr, mew])
                else:
                    collection = np.vstack((collection, [mass, bf_gamma, subhalo_kmmdsm.rb]))
                    col_info = np.vstack((col_info, [shape, mss, clr, mew]))
                tot_count += 1

        avg_gam, medi_gam, avg_rb, medi_rb = [np.mean(collection[:, 1]), np.median(collection[:, 1]),
                                              np.mean(collection[:, 2]), np.median(collection[:, 2])]
        if plot:
            fig = plt.figure(figsize=(8., 6.))
            ax = plt.gca()
            ax.set_xscale("log")
            pl.xlim([10 ** -2, 2.])
            pl.ylim([-0.02, 2.])
            for i,col in enumerate(collection):

                plt.plot(col[2], col[1], col_info[i, 0], ms=float(col_info[i, 1]), markerfacecolor='None',
                         markeredgecolor=col_info[i, 2], mew=float(col_info[i, 3]))
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
            plt.text(.012, 0.9, r'$\gamma$ Mean, Median: {:.2f},{:.2f}'.format(avg_gam, medi_gam),
                     fontsize=10, ha='left', va='center', color='Black')
            plt.text(.012, 0.8, r'$R_b$ Mean, Median: {:.2f},{:.2f}'.format(avg_rb, medi_rb),
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
        else:
            return collection

    def equipartition(self, x=np.array([10., 100.]), bnd_num=5):
        x = np.sort(x)
        a = [x[i:i + len(x) / bnd_num] for i in range(0, len(x), len(x) / bnd_num)]
        return a

    def multi_KKDSM_plot(self,
                         mass_list=[10.**5., 10.**6., 6.*10**6., 2.*10**7., 7.*10**7., 10**10.],
                         gcd_list=[0., 100.],
                         m_num=5, gcd_num=1, del_v_num=1, ms=2, tag='_0to100kpc'):
        color_list = ['y', 'c', 'm', 'r', 'g', 'b', 'k']
        shape_list = ['o', '+', '*', 'd', 'v', 's', 'h']
        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        pl.xlim([10 ** -2, 3.])
        pl.ylim([-0.02, 3.])

        d_vmax_min = 0.
        d_vmax_max = 1.
        if mass_list == None:
            mass_list, gcd_list = self.equipartition(m_num=4, gcd_num=1, del_v_num=1)
        bndry = np.zeros(len(mass_list), dtype=int)
        av_array = np.zeros((len(mass_list) - 1) * 4).reshape((len(mass_list) - 1, 4))
        leg_top = 3.
        for m in range(len(mass_list) - 1):
            if m == 0:
                print mass_list[-(m + 2)], mass_list[-(m+1)]
                slice = self.KMMDSM_fit(m_min=mass_list[-(m + 2)], m_max=mass_list[-(m+1)],
                                        gcd_min=gcd_list[0], gcd_max=gcd_list[-1], plot=False)
                print 'Halos added: ', len(slice)

            else:
                hold = self.KMMDSM_fit(m_min=mass_list[-(m + 2)], m_max=mass_list[-(m+1)],
                                        gcd_min=gcd_list[0], gcd_max=gcd_list[-1], plot=False)
                print 'Halos added: ', len(hold)
                slice = np.vstack((slice, hold))

            bndry[m + 1] = len(slice[:, 0])
            plt.plot(slice[bndry[m]:bndry[m+1]][:, 2], slice[bndry[m]:bndry[m+1]][:, 1],
                     shape_list[-(1 + m)], mfc='None', mec=color_list[-(1 + m)])
            leg_top -= 0.1
            plt.text(.012, leg_top, r'Mass Range [{:.2e} {:.2e}]'.format(mass_list[-(2 + m)], mass_list[-(1 + m)]),
                     fontsize=10, ha='left', va='center', color=color_list[-(1 + m)])
            leg_top -= 0.1
            av_array[m] = [np.mean(slice[bndry[m]:bndry[m+1]][:, 1]),
                           np.mean(slice[bndry[m]:bndry[m+1]][:, 2]),
                           np.median(slice[bndry[m]:bndry[m+1]][:, 1]),
                           np.median(slice[bndry[m]:bndry[m+1]][:, 2])]

            plt.text(.012, leg_top, r'$\gamma [R_b]$ Mean, Median: ' +
                     '{:.2f} {:.2f} [{:.2f} {:.2f}]'.format(av_array[m, 0], av_array[m, 1],
                                                            av_array[m, 2], av_array[m, 3]) +
                     ' Ntot: ' + str(bndry[m+1]-bndry[m]),
                     fontsize=10, ha='left', va='center', color=color_list[-(1 + m)])
        fig_name = self.dir + 'Joint_Sim_Plots/KMMDSM_Fit_mass_equipart_split_num_' + \
                   str(m_num) + tag + '.pdf'

        pl.xlabel(r'$R_b$   [kpc]', fontsize=20)
        pl.ylabel(r'$\gamma$', fontsize=20)
        fig.set_tight_layout(True)
        pl.savefig(fig_name)
        return

    def fit_hisogram(self, mlow=10**4., mhigh=10.**7.):
        gcd_range = np.linspace(0., 300., 100)

        for i in range(len(self.class_list)):
            print self.names[i]
            sub_o_int = self.class_list[i].find_subhalos(min_mass=mlow, max_mass=mhigh,
                                                         gcd_min=gcd_range[0],
                                                         gcd_max=gcd_range[-1],
                                                         print_info=False)
            if i == 0:
                subhalos = self.subhalos[i][sub_o_int][:, :6]
            else:
                subhalos = np.vstack((subhalos, self.subhalos[i][sub_o_int][:, :6]))

        print 'Making Number Density Histogram...'
        plt.hist(subhalos[:, 1], bins=gcd_range, log=True, normed=False,
                 weights=np.ones(subhalos[:, 1].size) / (subhalos[:, 1]**3. * 4. * np.pi / 3.),
                 alpha=0.3, color='Blue')

        pl.xlim([gcd_range[0], gcd_range[-1]])
        plt.xlabel(r'GC Distance (kpc)', fontsize=16)
        plt.ylabel(r'dN/dV', fontsize=16)

        plt.text(270, 10. ** -3.,
                 r'Mass Range: [{:.1e}, {:.1e}] $M_\odot$'.format(mlow, mhigh),
                 fontsize=10, ha='right', va='center', color='Black')

        fname = 'Number_Histogram_Func_GCD__Mass_Range_{:.1e}_{:.1e}'.format(mlow, mhigh) +\
            '.pdf'
        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)

        print 'Making Mass Density Histogram...'
        plt.hist(subhalos[:, 1], bins=gcd_range, log=True, normed=False,
                 weights=np.ones(subhalos[:, 1].size) * subhalos[:, 5] /
                         (subhalos[:, 1] ** 3. * 4. * np.pi / 3.),
                 alpha=0.3, color='Blue')

        pl.xlim([gcd_range[0], gcd_range[-1]])
        plt.xlabel(r'GC Distance (kpc)', fontsize=16)
        plt.ylabel(r'$M \frac{dN}{dV}$', fontsize=16)

        plt.text(270, 10. ** -3.,
                 r'Mass Range: [{:.1e}, {:.1e}] $M_\odot$'.format(mlow, mhigh),
                 fontsize=10, ha='right', va='center', color='Black')

        fname = 'Number_Mass_Histogram_Func_GCD__Mass_Range_{:.1e}_{:.1e}'.format(mlow, mhigh) + \
                '.pdf'
        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)

        print 'Calculating Fits...'
        for j, sub in enumerate(subhalos):
            rvmax, vmax, mass = [sub[4], sub[3], sub[5]]
            try:
                bf_gamma = find_gamma(mass, vmax, rvmax)
                subhalo_kmmdsm = KMMDSM(mass, bf_gamma, arxiv_num=160106781,
                                        vmax=vmax, rmax=rvmax)
                try:
                    gamma_plt = np.vstack((gamma_plt, [sub[1], bf_gamma]))
                    rb_plt = np.vstack((rb_plt, [sub[1], subhalo_kmmdsm.rb]))
                except UnboundLocalError:
                    gamma_plt = np.array([sub[1], bf_gamma])
                    rb_plt = np.array([sub[1], subhalo_kmmdsm.rb])
            except ValueError:
                bf_gamma = 0.

        print 'Finding fit to gerneralized normal distribution...'
        regions = 4
        r_groups = self.equipartition(x=gamma_plt[:, 0], bnd_num=regions)
        r_bounds = [r_groups[a][0] for a in range(regions)]
        r_bounds.append(r_groups[-1][-1])
        print 'GCD Bounds: ', r_bounds
        r_diff_tab = np.diff(r_bounds) / 2. + r_bounds[:-1]
        fit_tabs_g = np.zeros(regions * 7).reshape((regions, 7))
        fit_tabs_rb = np.zeros(regions * 7).reshape((regions, 7))

        for i in range(regions):
            print 'Dist Range: ', r_bounds[i], r_bounds[i+1]
            args_o_int = (r_bounds[i] < gamma_plt[:, 0]) & (gamma_plt[:, 0] < r_bounds[i + 1])
            print 'Number Subhalos in Range: ', args_o_int.sum()
            print 'Mean Gamma of Subhalos in Range: ', np.mean(gamma_plt[args_o_int][:, 1])

            hist_g, bin_edges_g = np.histogram(gamma_plt[args_o_int][:, 1], bins='auto', normed=True)
            x_fit_g = np.diff(bin_edges_g) / 2. + bin_edges_g[:-1]
            mu_g, sig_g, k_g = gen_norm_fit_finder(x_fit_g, hist_g, sigma=np.sqrt(hist_g), mu_low=.5,
                                                   mu_high=2., klow=.01, khigh=1., slow=.1, shigh=1.)
            median_g = mu_g
            std_dev_g_low = lower_sigma_gen_norm(mu_g, sig_g, k_g)
            std_dev_g_high = upper_sigma_gen_norm(mu_g, sig_g, k_g)
            fit_tabs_g[i] = [r_diff_tab[i], mu_g, sig_g, k_g, median_g,
                             std_dev_g_low, std_dev_g_high]
            # fig = plt.figure(figsize=(8., 6.))
            # ax = plt.gca()
            # x_lin = np.linspace(gamma_plt[args_o_int][:, 1].min(), gamma_plt[args_o_int][:, 1].max(), 100)
            # y_test = gen_norm_dist(x_lin, mu_g, sig_g, k_g)
            # plt.hist(gamma_plt[args_o_int][:, 1], bins='auto', normed=True, fc=None, ec='Black')
            # plt.plot(x_lin, y_test)
            # plt.axvline(x=fit_tabs_g[i,4] - fit_tabs_g[i,5], ymin=0, ymax=1, color='r')
            # plt.axvline(x=fit_tabs_g[i, 4] + fit_tabs_g[i, 6], ymin=0, ymax=1, color='r')
            # name = '/Users/SamWitte/Desktop/Gamma_Fit_Distribution_' + str(i) + '_.pdf'
            # pl.savefig(name)

            hist_rb, bin_edges_rb = np.histogram(rb_plt[args_o_int][:, 1], bins='auto', normed=True)
            x_fit_rb = np.diff(bin_edges_rb) / 2. + bin_edges_rb[:-1]
            pars, cov = curve_fit(burr_dist, x_fit_rb, hist_rb,  bounds=(0., np.inf))
            a, b, c = [pars[0], pars[1], pars[2]]
            med_rb = a * (2. ** (1./ c) - 1.) ** (-1. / b)
            std_dev_r_high = upper_sigma_burr(a, b, c, maxval=np.max(rb_plt[args_o_int][:, 1]))
            std_dev_r_low = lower_sigma_burr(a, b, c, minval=np.min(rb_plt[args_o_int][:, 1]))
            fit_tabs_rb[i] = [r_diff_tab[i], a, b, c, med_rb, std_dev_r_low, std_dev_r_high]
            # fig = plt.figure(figsize=(8., 6.))
            # ax = plt.gca()
            # x_lin = np.linspace(rb_plt[args_o_int][:, 1].min(), rb_plt[args_o_int][:, 1].max(), 100)
            # y_test = burr_dist(x_lin, pars[0], pars[1], pars[2])
            # plt.hist(rb_plt[args_o_int][:, 1], bins='auto', normed=True, color='White')
            # plt.plot(x_lin, y_test)
            # plt.axvline(x=fit_tabs_rb[i,4] - fit_tabs_rb[i,5], ymin=0, ymax=1, color='r')
            # plt.axvline(x=fit_tabs_rb[i, 4] + fit_tabs_rb[i, 6], ymin=0, ymax=1, color='r')
            # name = '/Users/SamWitte/Desktop/Rb_Fit_Distribution_BURR_' + str(i) + '_.pdf'
            # pl.savefig(name)

        print 'Gamma Fit Tabs: ', fit_tabs_g
        print 'Rb Fit Tabs: ', fit_tabs_rb

        print 'Making Gamma vs GCD scatter...'
        gam_mean_fit = np.polyfit(np.log10(fit_tabs_g[:, 0]), np.log10(fit_tabs_g[:, 4]), 1)
        gcd_tab_plt = np.logspace(1., np.log10(300.), 100)
        gam_fit_plt = 10. ** np.polyval(gam_mean_fit, np.log10(gcd_tab_plt))

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10, 300.])
        ymin = 0.3
        ymax = np.max(gamma_plt[:, 1])
        pl.ylim([ymin, ymax])
        plt.xlabel('GC Distance (kpc)', fontsize=16)
        plt.ylabel(r'$\gamma$', fontsize=16)
        fname = 'Gamma_vs_GCD_Scatter__Mass_Range_{:.1e}_{:.1e}'.format(mlow, mhigh) +\
            '.pdf'
        plt.plot(gamma_plt[:, 0], gamma_plt[:, 1], 'ro', alpha=0.3)
        plt.plot(fit_tabs_g[:, 0], fit_tabs_g[:, 4], 'kx', ms=6, mew=3)

        for x in range(fit_tabs_g[:, 0].size):
            if (fit_tabs_g[x, 4] - fit_tabs_g[x, 5]) > 0.:
                yerr_l = (np.log10(fit_tabs_g[x, 4] - fit_tabs_g[x, 5]) -
                          np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            else:
                yerr_l = 0.
            yerr_h = (np.log10(fit_tabs_g[x, 4] + fit_tabs_g[x, 6]) -
                      np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))

            plt.axvline(x=fit_tabs_g[x, 0], ymin=yerr_l, ymax=yerr_h,
                        linewidth=2, color='Black')

        gam_err_fit = np.polyfit(np.log10(fit_tabs_g[:, 0]), np.log10(fit_tabs_g[:, 4] + fit_tabs_g[:, 6]), 1)
        gam_local_err = 10. ** np.polyval(gam_err_fit, np.log10(8.5))
        gam_err_fit2 = np.polyfit(np.log10(fit_tabs_g[:, 0]), np.log10(fit_tabs_g[:, 4] - fit_tabs_g[:, 5]), 1)
        gam_local_err2 = 10. ** np.polyval(gam_err_fit2, np.log10(8.5))

        plt.plot(gcd_tab_plt, gam_fit_plt, 'k', ms=2)
        plt.text(12., 1.6, r'$\gamma(R_\oplus)$ = {:.3f}'.format(10. ** np.polyval(gam_mean_fit,np.log10(8.5))),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(12., 1.3, r'$\gamma^+(R_\oplus)$ = {:.3f}'.format(gam_local_err),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(12., 1.1, r'$\gamma^-(R_\oplus)$ = {:.3f}'.format(gam_local_err2),
                 fontsize=10, ha='left', va='center', color='Black')

        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)

        print 'Making Rb vs GCD scatter...'
        rb_mean_fit = np.polyfit(np.log10(fit_tabs_rb[:, 0]), np.log10(fit_tabs_rb[:, 4]), 1)
        rb_fit_plt = 10. ** np.polyval(rb_mean_fit, np.log10(gcd_tab_plt))

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10, 300.])
        ymin = np.min(rb_plt[:, 1])
        ymax = np.max(rb_plt[:, 1])
        pl.ylim([ymin, ymax])
        plt.xlabel('GC Distance (kpc)', fontsize=16)
        plt.ylabel(r'$R_b$', fontsize=16)
        fname = 'Rb_vs_GCD_Scatter__Mass_Range_{:.1e}_{:.1e}'.format(mlow, mhigh) + \
                '.pdf'
        plt.plot(rb_plt[:, 0], rb_plt[:, 1], 'ro', alpha=0.3)
        plt.plot(fit_tabs_rb[:, 0], fit_tabs_rb[:, 4], 'kx', ms=6, mew=3)
        plt.plot(gcd_tab_plt, rb_fit_plt, 'k', ms=2)

        for x in range(fit_tabs_rb[:, 0].size):
            if (fit_tabs_rb[x, 4] - fit_tabs_rb[x, 5]) > 0.:
                yerr_l = (np.log10(fit_tabs_rb[x, 4] - fit_tabs_rb[x, 5]) -
                          np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            else:
                yerr_l = 0.
            yerr_h = (np.log10(fit_tabs_rb[x, 4] + fit_tabs_rb[x, 6]) -
                      np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            plt.axvline(x=fit_tabs_rb[x, 0], ymin=yerr_l, ymax=yerr_h,
                        linewidth=2, color='Black')

        rb_err_fit = np.polyfit(np.log10(fit_tabs_rb[:, 0]), np.log10(fit_tabs_rb[:, 4] + fit_tabs_rb[:, 6]), 1)
        rb_err_fit2 = np.polyfit(np.log10(fit_tabs_rb[:, 0]), np.log10(fit_tabs_rb[:, 4] - fit_tabs_rb[:, 5]), 1)
        rb_local_err = 10. ** np.polyval(rb_err_fit, np.log10(8.5))
        rb_local_err2 = 10. ** np.polyval(rb_err_fit2, np.log10(8.5))

        legtot = ymax - ymin
        legtop = .8 * (ymax - ymin)
        down = legtot / 10
        plt.text(12., legtop, r'$R_b(R_\oplus)$ = {:.3f}'.format(10. ** np.polyval(rb_mean_fit, np.log10(8.5))),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(12., legtop - down, r'$R_b^+(R_\oplus)$ = {:.3f}'.format(rb_local_err),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(12., legtop - 2. * down, r'$R_b^-(R_\oplus)$ = {:.3f}'.format(rb_local_err2),
                 fontsize=10, ha='left', va='center', color='Black')
        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)

        return

    def fit_params(self):
        mass_list = np.array([4.10e+05, 5.35e+06, 1.70e+07, 3.16e+07, 5.52e+07, 1.18e+08, 1.e+10])
        mass_plt = np.diff(mass_list) / 2. + mass_list[:-1]
        med_gamma = np.array([0.896, 0.815, 0.828, 0.941, 1.017, 1.173])
        high_gamma = np.array([1.245, 1.234, 1.270, 1.258, 1.396, 1.494])
        low_gamma = np.array([0.453, 0.224, 0.202, 0.347, 0.481, 0.851])
        med_rb = np.array([ 0.050, 0.118, 0.228, 0.287, 0.317, 0.614])
        high_rb = np.array([ 0.091, 0.170, 0.366, 0.485, 0.526, 1.291])
        low_rb = np.array([ 0.025, 0.093, 0.160, 0.174, 0.207, 0.361])

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        xmax = 10. ** 10.
        xmin = 4. * 10 ** 5.
        pl.xlim([xmin, xmax])
        ymin = 0.1
        ymax = 2.
        pl.ylim([ymin, ymax])
        plt.xlabel(r'Subhalo Mass  [$M_\odot$]', fontsize=16)
        plt.ylabel(r'$\gamma (r_\oplus)$', fontsize=16)
        fname = 'Fit_Parameters_Gamma_@Earth_With_Mass.pdf'
        sub_m = np.array([])
        for i in range(len(self.subhalos)):
            sub_m = np.append(sub_m, self.subhalos[i][:, 5])
        median_mass = np.zeros(len(mass_list) - 1)
        for i in range(median_mass.size):
            median_mass[i] = np.median(sub_m[(sub_m > mass_list[i]) & (sub_m < mass_list[i + 1])])

        for i, m in enumerate(median_mass):
            yerr_l = np.log10(low_gamma[i] / ymin) / np.log10(ymax / ymin)
            yerr_h = np.log10(high_gamma[i] / ymin) / np.log10(ymax / ymin)
            xerr_l = np.log10(mass_list[i] / xmin) / np.log10(xmax / xmin)
            xerr_h = np.log10(mass_list[i + 1] / xmin) / np.log10(xmax / xmin)
            plt.axvline(x=m, ymin=yerr_l, ymax=yerr_h, linewidth=1.5, color='Red')
            plt.axhline(y=med_gamma[i], xmin=xerr_l, xmax=xerr_h, linewidth=1.5, color='Red')

        mtab = np.logspace(4, 10, 100)
        g_fit = np.mean(med_gamma)
        g_fit_up = np.mean(high_gamma)
        g_fit_down = np.mean(low_gamma)

        plt.axhline(y=g_fit, linewidth=1, color='Black')
        plt.axhline(y=g_fit_up, linewidth=1, color='Blue', ls='-.')
        plt.axhline(y=g_fit_down, linewidth=1, color='Blue', ls='-.')
        plt.text(5. * 10 ** 9., 10**-.45, r'$<\gamma> = {:.3f}$'.format(g_fit), fontsize=12, ha='right')
        plt.text(5. * 10 ** 9., 10**-.53, r'$\gamma^+ = {:.3f}$'.format(g_fit_up), fontsize=12, ha='right')
        plt.text(5. * 10 ** 9., 10**-.61, r'$\gamma^- = {:.3f}$'.format(g_fit_down), fontsize=12, ha='right')

        fig.set_tight_layout(True)
        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)

        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        xmax = 10. ** 10.
        xmin = 4. * 10 ** 5.
        pl.xlim([xmin, xmax])
        ymin = 0.01
        ymax = 1.5
        pl.ylim([ymin, ymax])
        plt.xlabel(r'Subhalo Mass  [$M_\odot$]', fontsize=16)
        plt.ylabel(r'$R_b (r_\oplus)$  [kpc]', fontsize=16)
        fname = 'Fit_Parameters_Rb_@Earth_With_Mass.pdf'

        rb_fit = np.polyfit(np.log10(median_mass), np.log10(med_rb), 1)
        rb_fit_up = np.polyfit(np.log10(median_mass), np.log10(high_rb), 1)
        rb_fit_down = np.polyfit(np.log10(median_mass), np.log10(low_rb), 1)
        mtab = np.logspace(4, 10, 100)
        rb_fit_plot = 10. ** np.polyval(rb_fit, np.log10(mtab))
        rb_fit_up_plot = 10. ** np.polyval(rb_fit_up, np.log10(mtab))
        rb_fit_low_plot = 10. ** np.polyval(rb_fit_down, np.log10(mtab))
        plt.plot(mtab, rb_fit_plot, linewidth=1, color='Black')
        plt.plot(mtab, rb_fit_up_plot, '-.', linewidth=1, color='Blue')
        plt.plot(mtab, rb_fit_low_plot, '-.', linewidth=1, color='Blue')
        plt.text(6 * 10 ** 5., 10 ** 0.,
                 r'$R_b(M) = 10^{{{:.3f}}} \times \left(\frac{{M}}{{M_\odot}}\right)^{{{:.3f}}}$'.format(rb_fit[1],
                                                                                                         rb_fit[0]),
                 fontsize=12)
        plt.text(6 * 10 ** 5., 10. ** -0.2,
                 r'$R_b^{{cons}}(M) = 10^{{{:.3f}}} \times \left(\frac{{M}}{{M_\odot}}\right)^{{{:.3f}}}$'.format(rb_fit_up[1],
                                                                                                                  rb_fit_up[0]),
                 fontsize=12, color='Blue')
        plt.text(6 * 10 ** 5., 10. ** -0.4,
                 r'$R_b^{{opt}}(M) = 10^{{{:.3f}}} \times \left(\frac{{M}}{{M_\odot}}\right)^{{{:.3f}}}$'.format(
                     rb_fit_down[1],
                     rb_fit_down[0]),
                 fontsize=12, color='Blue')

        for i, m in enumerate(median_mass):
            yerr_l = np.log10(low_rb[i] / ymin) / np.log10(ymax / ymin)
            yerr_h = np.log10(high_rb[i] / ymin) / np.log10(ymax / ymin)
            xerr_l = np.log10(mass_list[i] / xmin) / np.log10(xmax / xmin)
            xerr_h = np.log10(mass_list[i + 1] / xmin) / np.log10(xmax / xmin)
            plt.axvline(x=m, ymin=yerr_l, ymax=yerr_h, linewidth=1.5, color='Red')
            plt.axhline(y=med_rb[i], xmin=xerr_l, xmax=xerr_h, linewidth=1.5, color='Red')
        fig.set_tight_layout(True)
        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)

        return

    def obtain_number_density(self, min_mass=4.10e+05, max_mass=1.e+10):
        gcd_range = np.array([0., 30.])
        for i in range(len(self.class_list)):
            print self.names[i]
            sub_o_int = self.class_list[i].find_subhalos(min_mass=min_mass, max_mass=max_mass,
                                                         gcd_min=gcd_range[0],
                                                         gcd_max=gcd_range[-1],
                                                         print_info=False)
            if i == 0:
                subhalos = self.subhalos[i][sub_o_int][:, :6]
            else:
                subhalos = np.vstack((subhalos, self.subhalos[i][sub_o_int][:, :6]))

        mass_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 20)
        print 'Making Subhalo Number Density Histogram...'
        plt.hist(subhalos[:, 5], bins=mass_bins, log=True, normed=False,
                 weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.),
                 color='White')
        pl.gca().set_xscale("log")
        pl.xlim([min_mass, max_mass])
        plt.xlabel(r'Subhalo Mass [$M_\odot$]', fontsize=16)
        plt.ylabel(r'$\frac{dN}{dM dV}$', fontsize=16)

        fname = 'Subhalo_Number_Density_Mass_Range_{:.1e}_{:.1e}'.format(min_mass, max_mass) + \
                'GCD_range_{:.1f}_{:.1f}'.format(gcd_range[0], gcd_range[1]) + '.pdf'
        pl.savefig(self.dir + '/Joint_Sim_plots/' + fname)
        return


def gen_norm_dist(x, mu, sigma, k):
    try:
        ret_arr = np.zeros(len(x))
    except TypeError:
        x = np.array([x])
        ret_arr = np.zeros(len(x))
    for i in range(len(x)):
        if k == 0:
            y = (x[i] - mu) / sigma
            ret_arr[i] = np.exp(- y ** 2. / 2.) / \
                         (np.sqrt(2. * np.pi) * (sigma - k * (x[i] - mu)))
        elif (k > 0 and x[i] < mu + sigma / k) or (k < 0 and x[i] > mu + sigma / k):
            y = -1. / k * np.log(1. - k * (x[i] - mu) / sigma)
            ret_arr[i] = np.exp(- y ** 2. / 2.) / \
                         (np.sqrt(2. * np.pi) * (sigma - k * (x[i] - mu)))
        else:
            ret_arr[i] = np.inf
    return ret_arr

def lower_sigma_gen_norm(mu, sigma, k):
    if k < 0.:
        bndlow = mu + sigma / k
    else:
        bndlow = 10 ** -3.
    bounds = [(bndlow, mu)]
    test_list = np.linspace(bndlow, mu - 0.001, 50)
    int_list = np.zeros(len(test_list))
    for i, t in enumerate(test_list):
        int_list[i] = np.abs(quad(gen_norm_dist, t, mu, args=(mu, sigma, k))[0] - 0.34)
    lsig = test_list[np.argmin(int_list)]
    #lsig = minimize(lambda y: np.abs(quad(gen_norm_dist, y, mu, args=(mu, sigma, k))[0] - 0.34),
    #                np.array([mu - .1]), method='SLSQP', bounds=bounds, tol=10**-3.)
    print 'Lower Sigma [check]', quad(gen_norm_dist, lsig, mu, args=(mu, sigma, k))[0]
    return mu - lsig

def upper_sigma_gen_norm(mu, sigma, k):
    if k > 0.:
        bndup = mu + sigma / k
    else:
        bndup = np.inf
    bounds = [(mu, bndup)]
    test_list = np.logspace(np.log10(mu + 0.001), np.log10(bndup), 50)
    int_list = np.zeros(len(test_list))
    for i, t in enumerate(test_list):
        int_list[i] = np.abs(quad(gen_norm_dist, mu, t, args=(mu, sigma, k))[0] - 0.34)
    usig = test_list[np.argmin(int_list)]
    #usig = minimize(lambda y: np.abs(quad(gen_norm_dist, mu, y, args=(mu, sigma, k))[0] - 0.34),
    #                np.array([mu + .1]), method='SLSQP', bounds=bounds, tol=10**-3.)
    print 'Upper Sigma [check]', quad(gen_norm_dist, mu, usig, args=(mu, sigma, k))[0]
    return usig - mu


def gen_norm_fit_finder(x, y, sigma, mu_low=.01, mu_high=2.,
                        klow=0.1, khigh=1., slow=0.01, shigh=1.):

    def l_sqr(xd, xt, s):
        fix_in = s <= 0.
        s[fix_in] = 10 ** -4.
        return (((xd - xt) / s) ** 2.).sum()

    mu_tab = np.logspace(np.log10(mu_low), np.log10(mu_high), 50)
    k_tab = np.linspace(klow, khigh, 50)
    sig_tab = np.logspace(np.log10(slow), np.log10(shigh), 50)
    tot_num = mu_tab.size * k_tab.size * sig_tab.size

    full_tabs = np.zeros(tot_num * 4).reshape((tot_num, 4))
    count = 0
    for i, mu in enumerate(mu_tab):
        for j, s in enumerate(sig_tab):
            for z, k in enumerate(k_tab):
                theory = gen_norm_dist(x, mu, s, k)
                full_tabs[count, 3] = l_sqr(theory, y, sigma)
                full_tabs[count, 0] = mu
                full_tabs[count, 1] = s
                full_tabs[count, 2] = k
                count += 1
    arg = np.argmin(full_tabs[:, 3])
    bf_mu = full_tabs[arg, 0]
    bf_k = full_tabs[arg, 2]
    bf_sig = full_tabs[arg, 1]
    return bf_mu, bf_sig, bf_k


def beta_dist(x, b, c, d):
    ret_arr = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] >= b:
            ret_arr[i] = np.inf
        else:
            ret_arr[i] = b ** (1. - c - d) * x[i] ** (c - 1.) * (b - x[i]) ** (d - 1.) / beta(c, d)
    return ret_arr


def beta_fit_finder(x, y, sigma, clow=.1, chigh=10.,
                    dlow=.1, dhigh=10.):

    def l_sqr(xd, xt, s):
        fix_in = s <= 0.
        s[fix_in] = 10 ** -4.
        return (((xd - xt) / s) ** 2.).sum()

    b = np.max(x) + 10 ** -5.
    c_tab = np.linspace(clow, chigh, 30)
    d_tab = np.linspace(dlow, dhigh, 30)
    tot_num = c_tab.size * d_tab.size

    full_tabs = np.zeros(tot_num * 4).reshape((tot_num, 4))
    count = 0

    for j, c in enumerate(c_tab):
        for z, d in enumerate(d_tab):
            theory = beta_dist(x, b, c, d)
            full_tabs[count, 3] = l_sqr(theory, y, sigma)
            full_tabs[count, 0] = b
            full_tabs[count, 1] = c
            full_tabs[count, 2] = d
            count += 1
    arg = np.argmin(full_tabs[:, 3])
    bf_b = full_tabs[arg, 0]
    bf_d = full_tabs[arg, 2]
    bf_c = full_tabs[arg, 1]
    return bf_b, bf_c, bf_d


def burr_dist(x, a, b, c):
    try:
        dist = np.zeros(len(x))
    except TypeError:
        x = np.array([x])
        dist = np.zeros(len(x))
    valid_arg = x > 0.
    dist[valid_arg] = b * c / x[valid_arg] * ((x[valid_arg] / a) ** (b * c) /
                                              ((x[valid_arg] / a) ** b + 1.) ** (c + 1.))
    return dist


def lower_sigma_burr(a, b, c, minval=10**-3.):
    med = a * (2. ** (1. / c) - 1.) ** (-1. / b)
    bounds = [(minval, med)]
    test_list = np.logspace(np.log10(minval), np.log10(med - 0.001), 50)
    int_list = np.zeros(len(test_list))
    for i, t in enumerate(test_list):
        int_list[i] = np.abs(quad(burr_dist, t, med, args=(a, b, c))[0] - 0.34)
    lsig = test_list[np.argmin(int_list)]
    #lsig = minimize(lambda y: np.abs(quad(burr_dist, y, med, args=(a, b, c))[0] - 0.34),
    #                np.array([med - 0.01]), method='SLSQP', bounds=bounds, tol=10**-3.)
    print 'Lower Sigma [check]', quad(burr_dist, lsig, med, args=(a, b, c))[0]
    return med - lsig


def upper_sigma_burr(a, b, c, maxval=10.):
    med = a * (2. ** (1./ c) - 1.) ** (-1. / b)
    bounds = [(med, maxval)]
    test_list = np.logspace(np.log10(med + 0.001), np.log10(maxval), 50)
    int_list = np.zeros(len(test_list))
    for i, t in enumerate(test_list):
        int_list[i] = np.abs(quad(burr_dist, med, t, args=(a, b, c))[0] - 0.34)
    usig = test_list[np.argmin(int_list)]
    #usig = minimize(lambda y: np.abs(quad(burr_dist, med, y, args=(a, b, c))[0] - 0.34),
    #                np.array([maxval]), method='SLSQP', bounds=bounds, tol=10**-3.)

    print 'Upper Sigma [check]', quad(burr_dist, med, usig, args=(a, b, c))[0]
    return usig - med

