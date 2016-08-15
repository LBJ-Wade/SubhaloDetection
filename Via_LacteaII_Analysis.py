# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

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


class Via_Lactea_II(object):

    def __init__(self, profile=0, alpha=0.16, c=None, truncate=False, arxiv_num=13131729):
        self.dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
        self.f_name = 'ViaLacteaII_Info.dat'
        self.profile = profile
        self.truncate = truncate
        self.alpha = alpha
        self.arxiv_num = arxiv_num
        profile_list = ['einasto', 'nfw']
        self.profile_name = profile_list[self.profile]
        self.sv_prefix = 'VLII_Comparison_' + self.profile_name + '_Truncate_' +\
            str(self.truncate) + '_arXiv_num_' + str(arxiv_num) + '/'
        ensure_dir(self.dir + self.sv_prefix)
        self.c = c
        self.mass_part = 4.1 * 10 ** 3.

    def kill_unuseable_halos(self):
        via_lac_file = np.loadtxt(self.dir + self.f_name)
        for i, f in enumerate(via_lac_file):
            pass

    def find_subhalos(self, min_mass=4.1 * 10 ** 3., max_mass=10. ** 12., gcd_min=0., gcd_max=4000.,
                      print_info=True):
        via_lac_file = np.loadtxt(self.dir + self.f_name)
        list_of_interest = []
        for i,f in enumerate(via_lac_file):
            if (min_mass < f[5] < max_mass) and (gcd_min < f[1] < gcd_max):
                list_of_interest.append(i)
        print 'Mass Range: ', [min_mass, max_mass]
        print 'GC Distance Range: ', [gcd_min, gcd_max]
        if print_info:
            print 'Subhalos Matching: ', list_of_interest
            return
        else:
            return list_of_interest

    def Profile_Comparison(self):
        via_lac_file = np.loadtxt(self.dir + self.f_name)
        # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
        #                     rel_pos, rel_pos, M300, M600]                         (7-14)
        cmmnts = "{}   {}   {}   ".format('ID', 'Mass (SM)', 'R_vir (kpc)') + \
            "{}   {}   {}   ".format('R_s (kpc)', 'R_max_p (kpc)', 'M300 (SM)') + \
            "{}   {}   ".format('M600 (SM)', 'M_rmax (SM)') +\
            "{}   {}".format('Rho_max_sim', 'Rho_max_prof')
        profile_fit = np.zeros(len(via_lac_file) * 10).reshape((len(via_lac_file), 10))
        for i, sub in enumerate(via_lac_file):
            if self.truncate:
                mm = sub[5] / 0.005
            else:
                mm = sub[5]
            if self.profile == 0:
                self.subhalo = Einasto(mm, alpha=self.alpha, truncate=self.truncate, arxiv_num=self.arxiv_num)
            else:
                self.subhalo = NFW(mm, alpha=self.alpha, truncate=self.truncate, arxiv_num=self.arxiv_num)
            vmax, rmax = self.subhalo.vel_r_max()
            if self.subhalo.max_radius < 0.3:
                r3 = self.subhalo.max_radius
                r6 = self.subhalo.max_radius
            elif 0.3 < self.subhalo.max_radius < 0.6:
                r3 = 0.3
                r6 = self.subhalo.max_radius
            else:
                r3 = 0.3
                r6 = 0.6
            if self.subhalo.max_radius < sub[4]:
                r_of_max = self.subhalo.max_radius
            else:
                r_of_max = sub[4]
            profile_fit[i] = np.array([sub[0], sub[5], self.subhalo.virial_radius, self.subhalo.scale_radius,
                                       self.subhalo.max_radius, self.subhalo.int_over_density(r3),
                                       self.subhalo.int_over_density(r6), self.subhalo.int_over_density(r_of_max),
                                       sub[3] ** 2. / (4. * np.pi * newton_G * sub[4] ** 2.),
                                       vmax ** 2. / (4. * np.pi * newton_G * rmax ** 2.)])

        np.savetxt(self.dir + self.sv_prefix + 'Profile_Comparison.dat', profile_fit, fmt='%.5e', header=cmmnts)
        #  Output: [id, Mass (SM), R_vir (kpc), R_scale (kpc), R_max (kpc), M300 (SM), M600 (SM), M_rmax (SM)
        #  rho_r_max obs (SM/kpc^3), rho_r_max model (SM/kpc^3)]
        return

    def Density_Uncertanity(self):
        # TODO : Fix
        via_lac_file = np.loadtxt(self.dir + self.f_name)
        # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
        #                     rel_pos (3), rel_pos (3), M300, M600]                 (7-14)
        uncer_list = np.zeros(len(via_lac_file ) * 12).reshape((len(via_lac_file), 12))
        cmmnts = "{}   {}   {}   ".format('R300 (kpc)', 'M300 (SM)', 'Del M300 (SM)') +\
            "{}   {}   {}   ".format('R600 (kpc)', 'M600 (SM)', 'Del M600 (SM)') +\
            "{}   {}   {}   ".format('Rmax (kpc)', 'M_rmax (SM)', 'Del M_rmax (SM)') +\
            "{}   {}   {}".format('R_tot (kpc)', 'Mtot (SM)','Del Mtot (SM)')
        mass_per_p = 4.1 * 10 ** 3.
        for i, sub in enumerate(via_lac_file):
            m300 = np.array([0.3, sub[13], np.sqrt(sub[13] * mass_per_p)])
            m600 = np.array([0.6, sub[14], np.sqrt(sub[14] * mass_per_p)])
            m_rmax = np.array([sub[4], np.sqrt(sub[3] ** 2. * sub[4] / newton_G),
                               np.sqrt(sub[3] ** 2. * sub[4] * mass_per_p / newton_G)])
            mtot = np.array([sub[6], sub[5], np.sqrt(sub[5] * mass_per_p)])
            err_info = np.array([m300, m600, m_rmax, mtot])
            uncer_list[i] = err_info.flatten()
        np.savetxt(self.dir + '/VL_Subhalo_Uncertainty.dat', uncer_list, fmt='%.5e',header=cmmnts)
        return

    def goodness_of_fit(self):
        via_lac_file = np.loadtxt(self.dir + self.f_name)
        # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
        #                     rel_pos (3), rel_pos (3), M300, M600]                 (7-14)
        dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
        via_lac_err = np.loadtxt(dir + 'VL_Subhalo_Uncertainty.dat')
        # Output: [R300 (kpc), \Delta M300 (SM), \Delta rho300 (SM/kpc^3),              (0-2)
        # R600 (kpc), \Delta M600 (SM), \Delta rho600 (SM/kpc^3),                       (3-5)
        # R_max (kpc), \Delta M_rmax (SM), \Delta rho_rmax (SM/kpc^3), R_tot (kpc),     (6-9)
        # \Delta M_tot (SM), \Delta rho_tot (SM/kpc^3)]                                 (10-11)
        mass_per_p = 4.1 * 10 ** 3.
        gof = 0.
        num_obs = 0.
        n_fitted = 1.
        for i, sub in enumerate(via_lac_file):
            if self.truncate:
                mm = sub[5] / 0.005
            else:
                mm = sub[5]
            if self.profile == 0:
                self.subhalo = Einasto(mm, alpha=self.alpha, concentration_param=self.c,
                                       truncate=self.truncate, arxiv_num=self.arxiv_num)
            else:
                self.subhalo = NFW(mm, alpha=self.alpha, concentration_param=self.c,
                                   truncate=self.truncate, arxiv_num=self.arxiv_num)
            vmax, rmax = self.subhalo.vel_r_max()

            del_m_list = np.array([[0., 0., 0.], [0.3, sub[13], np.sqrt(sub[13] * mass_per_p)],
                                   [0.6, sub[14], np.sqrt(sub[14] * mass_per_p)],
                                   [sub[4], sub[3] ** 2. * sub[4] / newton_G,
                                    np.sqrt(sub[3] ** 2. * sub[4] * mass_per_p / newton_G)],
                                   [sub[6], sub[5], np.sqrt(sub[5] * mass_per_p)]])

            del_m_list = del_m_list[np.argsort(del_m_list[:, 0])]
            vr_pos = np.where(del_m_list[:, 0] == sub[6])[0][0]
            del_m_list = del_m_list[np.unique(del_m_list[:, 0], return_index=True)[1]]
            for k in range(len(del_m_list)):
                if del_m_list[k, 1] > del_m_list[vr_pos, 1]:
                    del_m_list[k, 1] = del_m_list[vr_pos, 1]

            model_list = np.zeros_like(del_m_list)
            model_list[:, 0] = del_m_list[:, 0]
            for k, r in enumerate(model_list[:, 0]):
                if k != 0.:
                    model_list[k, 1] = self.subhalo.int_over_density(r)
            m_diff = np.array([])
            model_diff = np.array([])

            # TODO: Consdier whether denisty rmax should be at fixed r or function of
            # rmax and vmax (how does one include this dependence)?

            for j in range(del_m_list[:, 0].size - 1):
                if j != 0:
                    m_diff = np.vstack((m_diff, [del_m_list[j + 1, 0],
                                                (del_m_list[j + 1, 1] - del_m_list[j, 1]),
                                                 (del_m_list[j + 1, 2] + del_m_list[j, 2])]))
                    model_diff = np.vstack((model_diff, [model_list[j + 1, 0],
                                                (model_list[j + 1, 1] - model_list[j, 1])]))
                else:
                    m_diff = np.append(m_diff, [del_m_list[j + 1, 0],
                                                 (del_m_list[j + 1, 1] - del_m_list[j, 1]),
                                                (del_m_list[j + 1, 2] + del_m_list[j, 2])])
                    model_diff = np.append(model_diff, [model_list[j + 1, 0],
                                                         (model_list[j + 1, 1] - model_list[j, 1])])

            rho_rmax = np.array([sub[4], sub[3] ** 2. / newton_G,
                                 np.sqrt(sub[3] ** 2. * mass_per_p / (newton_G * sub[4]))])
            model_rho_rmax = np.array([sub[4], self.subhalo.density(sub[4]) * sub[4]**2. * 4. * np.pi ])

            for ob in range(m_diff[:, 0].size):
                gof += ((m_diff[ob, 1] - model_diff[ob, 1]) / m_diff[ob, 2]) ** 2.
                num_obs += 1

            gof += ((rho_rmax[1] - model_rho_rmax[1]) / rho_rmax[2]) ** 2.
            num_obs += 1

        print 'Profile: ', self.profile_name
        print 'Truncate Profile: ', self.truncate
        print 'Parameterization from arXiv: ', self.arxiv_num
        print 'Degrees of Freedom: ', num_obs - n_fitted
        print 'Reduced Chi-squared TS: ', gof / (num_obs - n_fitted)


def plot_sample_comparison(sub_num=0, plot=True):

    color_list = ['Aqua', 'Magenta', 'Orange', 'Green', 'Red', 'Brown']
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ViaLacteaII_Info.dat'
    via_lac_file = np.loadtxt(dir + f_name)
    # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
    #                     rel_pos, rel_pos, M300, M600]                         (7-14)
    via_lac_err = np.loadtxt(dir + 'VL_Subhalo_Uncertainty.dat')
    mass_per_p = 4.1 * 10 ** 3.
    s_mass = via_lac_file[sub_num, 5]
    subhalo_e_t = Einasto(s_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_n_t = NFW(s_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_e_nt = Einasto(s_mass, 0.16, truncate=False, arxiv_num=13131729)
    subhalo_n_nt = NFW(s_mass, 0.16, truncate=False, arxiv_num=13131729)
    subhalo_e_bert = Einasto(s_mass, 0.16, truncate=False, arxiv_num=160106781)
    subhalo_n_bert = NFW(s_mass, 0.16, truncate=False, arxiv_num=160106781)

    num_d_pts = 100
    r_tab = np.logspace(-2., 1., num_d_pts)

    den_e_t = np.zeros_like(r_tab)
    den_n_t = np.zeros_like(r_tab)
    den_e_nt = np.zeros_like(r_tab)
    den_n_nt = np.zeros_like(r_tab)
    den_n_bert = np.zeros_like(r_tab)
    den_e_bert = np.zeros_like(r_tab)
    for i in range(num_d_pts):
        den_e_t[i] = subhalo_e_t.int_over_density(r_tab[i])
        den_n_t[i] = subhalo_n_t.int_over_density(r_tab[i])
        den_e_nt[i] = subhalo_e_nt.int_over_density(r_tab[i])
        den_n_nt[i] = subhalo_n_nt.int_over_density(r_tab[i])
        den_e_bert[i] = subhalo_e_bert.int_over_density(r_tab[i])
        den_n_bert[i] = subhalo_n_bert.int_over_density(r_tab[i])

    sub = via_lac_file[sub_num]
    del_m_list = np.array([[0., 0., 0.], [0.3, sub[13], np.sqrt(sub[13] * mass_per_p)],
                           [0.6, sub[14], np.sqrt(sub[14] * mass_per_p)],
                           [sub[4], sub[3] ** 2. * sub[4] / newton_G,
                            np.sqrt(sub[3] ** 2. * sub[4] * mass_per_p / newton_G)],
                           [sub[6], sub[5], np.sqrt(sub[5] * mass_per_p)]])

    del_m_list = del_m_list[np.argsort(del_m_list[:, 0])]
    vr_pos = np.where(del_m_list[:, 0] == sub[6])[0][0]
    del_m_list = del_m_list[np.unique(del_m_list[:, 0], return_index=True)[1]]
    for k in range(len(del_m_list)):
        if del_m_list[k, 1] > del_m_list[vr_pos, 1]:
            del_m_list[k, 1] = del_m_list[vr_pos, 1]
    plotpoints = np.zeros((len(del_m_list) - 1) * 3).reshape(((len(del_m_list) - 1), 3))
    plotpoints[:, 0] = del_m_list[1:, 0]
    plotpoints[:, 1] = np.diff(del_m_list[:, 1])
    for y in range(len(del_m_list[:, 2]) - 1):
        plotpoints[y, 2] = (del_m_list[y, 2] + del_m_list[y + 1, 2])
    if plot:
        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10 ** -2., 10.])
        ymax = 10. ** 8.
        ymin = 10 ** 2.
        pl.ylim([ymin, ymax])
        for x in range(del_m_list[:, 0].size - 1):
            plt.plot(del_m_list[x + 1, 0], del_m_list[x + 1, 1], 'o', ms=3, color='Black')
            try:
                yerr_l = (np.log10(del_m_list[x + 1, 1] - 3. * del_m_list[x + 1, 2]) -
                              np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            except:
                yerr_l = 0.
            plt.axvline(x=del_m_list[x + 1, 0],
                        ymin=yerr_l,
                        ymax=(np.log10(del_m_list[x + 1, 1] + 3. * del_m_list[x + 1, 2]) -
                              np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin)),
                        linewidth=1, color='Black')

        plt.plot(r_tab, den_e_t, lw=1, color=color_list[0], label='Ein (T)')
        plt.plot(r_tab, den_n_t, lw=1, color=color_list[1], label='NFW (T)')
        plt.plot(r_tab, den_e_nt, lw=1, color=color_list[2], label='Ein (NT)')
        plt.plot(r_tab, den_n_nt, lw=1, color=color_list[3], label='NFW (NT)')
        plt.plot(r_tab, den_e_bert, lw=1, color=color_list[4], label='Ein (Bertone)')
        plt.plot(r_tab, den_n_bert, lw=1, color=color_list[5], label='NFW (Bertone)')

        plt.text(2., 1. * 10 ** 4., 'Einasto (T)', fontsize=10, ha='left', va='center', color=color_list[0])
        plt.text(2., 1. * 10 ** 3.8, 'NFW (T)', fontsize=10, ha='left', va='center', color=color_list[1])
        plt.text(2., 1. * 10 ** 3.6, 'Einasto (NT)', fontsize=10, ha='left', va='center', color=color_list[2])
        plt.text(2., 1. * 10 ** 3.4, 'NFW (NT)', fontsize=10, ha='left', va='center', color=color_list[3])
        plt.text(2., 1. * 10 ** 3.2, 'Einasto (Bertone)', fontsize=10, ha='left', va='center', color=color_list[4])
        plt.text(2., 1. * 10 ** 3., 'NFW (Bertone)', fontsize=10, ha='left', va='center', color=color_list[5])

        plt.text(1.3 * 10**-2., 1. * 10 ** 7., r'Halo Mass: {:.1e} $M_\odot$'.format(via_lac_file[sub_num, 5]),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(1.3 * 10 ** -2., 1. * 10 ** 6.8, 'GC Dist {:.1f} kpc'.format(via_lac_file[sub_num, 1]),
                 fontsize=10, ha='left', va='center', color='Black')

        fig_name = dir + '/Via_Lac_plots/' + 'Subhalo_Number_' + str(sub_num) +\
            '_Profile_Comparison.pdf'

        pl.xlabel('Distance [kpc]', fontsize=20)
        pl.ylabel(r'M(r)  [$M_\odot$]', fontsize=20)
        fig.set_tight_layout(True)
        pl.savefig(fig_name)
        return
    else:
        return del_m_list, r_tab, den_e_t, den_n_t, den_e_nt, den_n_nt, den_e_bert, den_n_bert


def multi_slice(m_low=4.1 * 10 ** 3., m_high =10.**12., m_num=10, gc_d_min=0., gc_d_max=4000.,
                gc_d_num=2, plot=True, alpha=.1, ms=1, p_mtidal=False, p_mrmax=False,
                p_m300=False, p_m600=False):
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    color_list = ["Aqua", "Red", "Green", "Magenta", "Black"]
    shape_list = ['o', 'x']
    m_list = np.logspace(np.log10(m_low), np.log10(m_high), m_num)
    gcd_list = np.linspace(gc_d_min, gc_d_max, gc_d_num)
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** -2., 2 * 10.])
    ymax = 10. ** 9.
    ymin = 10 ** 2.
    pl.ylim([ymin, ymax])
    for g in range(gcd_list.size - 1):
        for m in range(m_list.size - 1):
            print [m_list[m], m_list[m+1]],[gcd_list[g],gcd_list[g+1]]
            collect_info = plot_slice(m_low=m_list[m], m_high =m_list[m + 1], gc_d_min=gcd_list[g],
                                      gc_d_max=gcd_list[g + 1], plot=False, alpha=.2, ms=2)
            for sub in collect_info:
                if p_m300:
                    plt.plot(sub[0], sub[1], 'o', ms=ms, color=color_list[m + g], alpha=alpha)
                if p_m600:
                    plt.plot(sub[2], sub[3], 'o', ms=ms, color=color_list[m + g], alpha=alpha)
                if p_mrmax:
                    plt.plot(sub[4], sub[5], 'o', ms=ms, color=color_list[m + g], alpha=alpha)
                if p_mtidal:
                    plt.plot(sub[6], sub[7], 'o', ms=ms, color=color_list[m + g], alpha=alpha)

    plot_lab = ''
    if p_mtidal:
        plot_lab += '_plot_mtidal'
    if p_mrmax:
        plot_lab += '_plot_mrmax'
    if p_m300:
        plot_lab += '_plot_m300'
    if p_m600:
        plot_lab += '_plot_m600'
    fig_name = dir + '/Via_Lac_plots/' + 'Subhalo_MassDistSplice_with_Mmin_{:.2e}'.format(m_low) + \
               '_Mhigh_{:.2e}'.format(m_high) + '_M_num_' + str(m_num) +\
               '_GCd_low_{:.1f}'.format(gc_d_min) + '_GCd_high_{:.1f}'.format(gc_d_max) +\
               '_GCd_num_' + str(gc_d_num) + plot_lab + '.pdf'

    pl.xlabel('Distance [kpc]', fontsize=20)
    pl.ylabel(r'M(r)  [$M_\odot$]', fontsize=20)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def plot_slice(m_low=4.1 * 10 ** 3., m_high =10.**12., gc_d_min=0., gc_d_max=200., plot=True,
               alpha=.1, ms=1):
    color_list = ['Aqua', 'Magenta', 'Orange', 'Green', 'Red', 'Brown']
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ViaLacteaII_Info.dat'
    via_lac_file = np.loadtxt(dir + f_name)
    # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
    #                     rel_pos, rel_pos, M300, M600]                         (7-14)
    via_lac_err = np.loadtxt(dir + 'VL_Subhalo_Uncertainty.dat')
    collection = np.array([])
    for i,sub in enumerate(via_lac_file):
        if m_low <= sub[5] <= m_high and gc_d_min <= sub[1] <= gc_d_max:
            try:
                collection = np.vstack((collection,
                                        np.array([0.3, sub[13], 0.6, sub[14], sub[4],
                                                  sub[3] ** 2. * sub[4] / newton_G,
                                                  sub[6], sub[5]])))
            except ValueError:
                collection = np.array([0.3, sub[13], 0.6, sub[14], sub[4],
                                       sub[3] ** 2. * sub[4] / newton_G,
                                       sub[6], sub[5]])

    if plot:
        fig = plt.figure(figsize=(8., 6.))
        ax = plt.gca()
        ax.set_xscale("log")
        ax.set_yscale('log')
        pl.xlim([10 ** -2., 10.])
        ymax = 10. ** 9.
        ymin = 10 ** 2.
        pl.ylim([ymin, ymax])
        max_m = np.max(collection[:, -1])
        min_m = np.min(collection[:, -1])
        for i, h in enumerate(collection):
            if i == 1:
                labels = ['M300(600)', r'$M_{R_{max}}$', r'$M_{tidal}$']
            else:
                labels = [None, None, None]
            plt.plot(h[0], h[1], 'o', ms=ms, color='Black', alpha = alpha, label=labels[0])
            plt.plot(h[2], h[3], 'o', ms=ms, color='Black', alpha = alpha)
            plt.plot(h[4], h[5], 'o', ms=ms, color='Red', alpha = alpha, label=labels[1])
            plt.plot(h[6], h[7], 'o', ms=ms, color='Blue', alpha = alpha, label=labels[2])

        subhalo_e_t_1 = Einasto(min_m / 0.005, 0.16, truncate=True, arxiv_num=13131729)
        subhalo_n_t_1 = NFW(min_m / 0.005, 0.16, truncate=True, arxiv_num=13131729)
        subhalo_e_nt_1 = Einasto(min_m, 0.16, truncate=False, arxiv_num=13131729)
        subhalo_n_nt_1 = NFW(min_m, 0.16, truncate=False, arxiv_num=13131729)
        subhalo_e_bert_1 = Einasto(min_m, 0.16, truncate=False, arxiv_num=160106781)
        subhalo_n_bert_1 = NFW(min_m, 0.16, truncate=False, arxiv_num=160106781)
        subhalo_e_t_2 = Einasto(max_m / 0.005, 0.16, truncate=True, arxiv_num=13131729)
        subhalo_n_t_2 = NFW(max_m / 0.005, 0.16, truncate=True, arxiv_num=13131729)
        subhalo_e_nt_2 = Einasto(max_m, 0.16, truncate=False, arxiv_num=13131729)
        subhalo_n_nt_2 = NFW(max_m, 0.16, truncate=False, arxiv_num=13131729)
        subhalo_e_bert_2 = Einasto(max_m, 0.16, truncate=False, arxiv_num=160106781)
        subhalo_n_bert_2 = NFW(max_m, 0.16, truncate=False, arxiv_num=160106781)

        num_d_pts = 100
        r_tab = np.logspace(-2., 1., num_d_pts)

        den_e_t = np.zeros_like(r_tab)
        den_n_t = np.zeros_like(r_tab)
        den_e_nt = np.zeros_like(r_tab)
        den_n_nt = np.zeros_like(r_tab)
        den_n_bert = np.zeros_like(r_tab)
        den_e_bert = np.zeros_like(r_tab)
        den_e_t2 = np.zeros_like(r_tab)
        den_n_t2 = np.zeros_like(r_tab)
        den_e_nt2 = np.zeros_like(r_tab)
        den_n_nt2 = np.zeros_like(r_tab)
        den_n_bert2 = np.zeros_like(r_tab)
        den_e_bert2 = np.zeros_like(r_tab)
        for i in range(num_d_pts):
            den_e_t[i] = subhalo_e_t_1.int_over_density(r_tab[i])
            den_n_t[i] = subhalo_n_t_1.int_over_density(r_tab[i])
            den_e_nt[i] = subhalo_e_nt_1.int_over_density(r_tab[i])
            den_n_nt[i] = subhalo_n_nt_1.int_over_density(r_tab[i])
            den_e_bert[i] = subhalo_e_bert_1.int_over_density(r_tab[i])
            den_n_bert[i] = subhalo_n_bert_1.int_over_density(r_tab[i])
            den_e_t2[i] = subhalo_e_t_2.int_over_density(r_tab[i])
            den_n_t2[i] = subhalo_n_t_2.int_over_density(r_tab[i])
            den_e_nt2[i] = subhalo_e_nt_2.int_over_density(r_tab[i])
            den_n_nt2[i] = subhalo_n_nt_2.int_over_density(r_tab[i])
            den_e_bert2[i] = subhalo_e_bert_2.int_over_density(r_tab[i])
            den_n_bert2[i] = subhalo_n_bert_2.int_over_density(r_tab[i])

        plt.plot(r_tab, den_e_t, '-.', r_tab, den_e_t2, '-.', lw=1, color=color_list[0], label='Ein (T)', alpha=0.5)
        plt.plot(r_tab, den_n_t, '-.', r_tab, den_n_t2, '-.', lw=1, color=color_list[1], label='NFW (T)', alpha=0.5)
        plt.plot(r_tab, den_e_nt, '-.', r_tab, den_e_nt2, '-.', lw=1, color=color_list[2], label='Ein (NT)', alpha=0.5)
        plt.plot(r_tab, den_n_nt, '-.', r_tab, den_n_nt2, '-.', lw=1, color=color_list[3], label='NFW (NT)', alpha=0.5)
        plt.plot(r_tab, den_e_bert, '-.', r_tab, den_e_bert2, '-.', lw=1, color=color_list[4], label='Ein (Bertone)', alpha=0.5)
        plt.plot(r_tab, den_n_bert, '-.', r_tab, den_n_bert, '-.', lw=1, color=color_list[5], label='NFW (Bertone)', alpha=0.5)

        plt.text(1.3 * 10 ** -2., 1. * 10 ** 8.7,
                 r'Halo Masses: [{:.1e}, {:.1e}] $M_\odot$'.format(m_low, m_high),
                 fontsize=10, ha='left', va='center', color='Black')

        plt.text(1.3 * 10 ** -2., 1. * 10 ** 8.4,
                 r'GC Dist: [{:.1e}, {:.1e}] kpc'.format(gc_d_min, gc_d_max),
                 fontsize=10, ha='left', va='center', color='Black')

        plt.legend(loc=4, fontsize=10)

        fig_name = dir + '/Via_Lac_plots/' + 'Subhalos_with_Mmin_{:.2e}'.format(m_low) +\
            '_Mhigh_{:.2e}'.format(m_high) + '_GCd_low_{:.1f}'.format(gc_d_min) +\
            '_GCd_high_{:.1f}'.format(gc_d_max) + '.pdf'

        pl.xlabel('Distance [kpc]', fontsize=20)
        pl.ylabel(r'M(r)  [$M_\odot$]', fontsize=20)
        fig.set_tight_layout(True)
        pl.savefig(fig_name)
    else:
        return collection


def catagorial_subplots(min_mass=4.1 * 10 ** 3., max_mass=10. ** 12., gcd_min=0., gcd_max=4000., n_plots=2):
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ViaLacteaII_Info.dat'
    via_lac_file = np.loadtxt(dir + f_name)
    list_of_interest = []
    color_list = ['Aqua', 'Magenta', 'Orange', 'Green', 'Red', 'Brown']
    for i, f in enumerate(via_lac_file):
        if (min_mass < f[5] < max_mass) and (gcd_min < f[1] < gcd_max):
            list_of_interest.append(i)
        if len(list_of_interest) == n_plots:
            break
    if len(list_of_interest) < n_plots:
        print 'Not this many subhalos matching parameters!'
        n_plots = len(list_of_interest)

    print 'Subhalo Numbers: ', n_plots
    ncol = 2
    nrows = n_plots / ncol
    f, ax = plt.subplots(nrows, ncol)
    pl.xlim([10 ** -2., 10.])
    ymax = 10. ** 8.
    ymin = 10 ** 2.
    j = 0
    numbers = '_'
    for i, halo in enumerate(list_of_interest):
        if i == nrows:
            j = 1
        ii = i % nrows
        del_m_list, r_tab, den_e_t, den_n_t, den_e_nt, \
            den_n_nt, den_e_bert, den_n_bert = plot_sample_comparison(sub_num=halo, plot=False)

        if nrows > 1:
            ax[ii, j].set_xscale("log")
            ax[ii, j].set_yscale('log')
            ax[ii, j].set_ylim([ymin, ymax])
        else:
            ax[j].set_xscale("log")
            ax[j].set_yscale('log')
            ax[j].set_ylim([ymin, ymax])
        for x in range(del_m_list[:, 0].size - 1):
            if nrows > 1:
                ax[ii, j].plot(del_m_list[x + 1, 0], del_m_list[x + 1, 1], 'o', ms=3, color='Black')
            else:
                ax[j].plot(del_m_list[x + 1, 0], del_m_list[x + 1, 1], 'o', ms=3, color='Black')
            try:
                yerr_l = (np.log10(del_m_list[x + 1, 1] - 3. * del_m_list[x + 1, 2]) -
                              np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            except:
                yerr_l = 0.
            if nrows > 1:
                ax[ii, j].axvline(x=del_m_list[x + 1, 0],
                                  ymin=yerr_l,
                                  ymax=(np.log10(del_m_list[x + 1, 1] + 3. *
                                                 del_m_list[x + 1, 2]) -
                                  np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin)),
                                  linewidth=1, color='Black')
            else:
                ax[j].axvline(x=del_m_list[x + 1, 0],
                              ymin=yerr_l,
                              ymax=(np.log10(del_m_list[x + 1, 1] + 3. * del_m_list[x + 1, 2]) -
                                    np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin)),
                              linewidth=1, color='Black')
        if nrows > 1:
            ax[ii, j].plot(r_tab, den_e_t, lw=1, color=color_list[0], label='Ein (T)')
            ax[ii, j].plot(r_tab, den_n_t, lw=1, color=color_list[1], label='NFW (T)')
            ax[ii, j].plot(r_tab, den_e_nt, lw=1, color=color_list[2], label='Ein (NT)')
            ax[ii, j].plot(r_tab, den_n_nt, lw=1, color=color_list[3], label='NFW (NT)')
            ax[ii, j].plot(r_tab, den_e_bert, lw=1, color=color_list[4], label='Ein (Bertone)')
            ax[ii, j].plot(r_tab, den_n_bert, lw=1, color=color_list[5], label='NFW (Bertone)')
        else:
            ax[j].plot(r_tab, den_e_t, lw=1, color=color_list[0], label='Ein (T)')
            ax[j].plot(r_tab, den_n_t, lw=1, color=color_list[1], label='NFW (T)')
            ax[j].plot(r_tab, den_e_nt, lw=1, color=color_list[2], label='Ein (NT)')
            ax[j].plot(r_tab, den_n_nt, lw=1, color=color_list[3], label='NFW (NT)')
            ax[j].plot(r_tab, den_e_bert, lw=1, color=color_list[4], label='Ein (Bertone)')
            ax[j].plot(r_tab, den_n_bert, lw=1, color=color_list[5], label='NFW (Bertone)')
        numbers += str(halo) + '_'
    f.set_tight_layout(True)

    fname = 'HaloSubplots_Numbers_' + numbers + '.pdf'
    print fname
    pl.savefig(dir + '/Via_Lac_plots/' + fname)
    return


def mass_ratio_vs_gcd(min_mass=10.**7., max_mass=2.*10**7., var1='M300', var2='Mtot'):
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ViaLacteaII_Info.dat'
    via_lac_file = np.loadtxt(dir + f_name)
    list_of_interest = []
    color_list = ['Aqua', 'Magenta', 'Orange', 'Green', 'Red', 'Brown']
    for i, f in enumerate(via_lac_file):
        if min_mass < f[5] < max_mass:
            list_of_interest.append(i)
    data_pts = np.array([])
    rmax_bnds = [4000., 0.]
    for i, sub in enumerate(list_of_interest):
        if var1 == 'M300' and var2 == 'M600':
            if via_lac_file[sub, 14] > via_lac_file[sub, 5]:
                m600 = via_lac_file[sub, 5]
            else:
                m600 = via_lac_file[sub, 14]
            m_ratio = np.min([via_lac_file[sub, 13] / m600, 1.])
        elif var1 == 'M300' and var2 == 'Mtot':
            m_ratio = np.min([via_lac_file[sub, 13] / (via_lac_file[sub, 5]), 1.])
        elif var1 == 'Mrmax' and var2 == 'Mtot':
            if (via_lac_file[sub, 3] ** 2. * via_lac_file[sub, 4] / newton_G) > via_lac_file[sub, 5]:
                mrmax = via_lac_file[sub, 5]
            else:
                mrmax = via_lac_file[sub, 3] ** 2. * via_lac_file[sub, 4] / newton_G
            m_ratio = mrmax / via_lac_file[sub, 5]
            rmax_bnds[0] = min(rmax_bnds[0], via_lac_file[sub, 4])
            rmax_bnds[1] = max(rmax_bnds[1], via_lac_file[sub, 4])
        else:
            print 'Mratio not defined!'
            m_ratio = 0.
        gcd = via_lac_file[sub, 1]
        if data_pts.size == 0:
            data_pts = np.array([gcd, m_ratio])
        else:
            data_pts = np.vstack((data_pts, [gcd, m_ratio]))
    if var1 == 'Mrmax':
        print 'Rmax lower and upper bounds: ', rmax_bnds

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    pl.xlim([10 ** 0., 4. * 10. ** 3.])
    ymax = 1.1
    ymin = 0.
    pl.ylim([ymin, ymax])
    for i in range(len(data_pts)):
        pl.plot(data_pts[i, 0], data_pts[i, 1], 'o', ms=1, color='Black')

    subhalo_e_t_1 = Einasto(min_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_n_t_1 = NFW(min_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_e_nt_1 = Einasto(min_mass, 0.16, truncate=False, arxiv_num=13131729)
    subhalo_n_nt_1 = NFW(min_mass, 0.16, truncate=False, arxiv_num=13131729)
    subhalo_e_bert_1 = Einasto(min_mass, 0.16, truncate=False, arxiv_num=160106781)
    subhalo_n_bert_1 = NFW(min_mass, 0.16, truncate=False, arxiv_num=160106781)
    subhalo_e_t_2 = Einasto(max_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_n_t_2 = NFW(max_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_e_nt_2 = Einasto(max_mass, 0.16, truncate=False, arxiv_num=13131729)
    subhalo_n_nt_2 = NFW(max_mass, 0.16, truncate=False, arxiv_num=13131729)
    subhalo_e_bert_2 = Einasto(max_mass, 0.16, truncate=False, arxiv_num=160106781)
    subhalo_n_bert_2 = NFW(max_mass, 0.16, truncate=False, arxiv_num=160106781)

    m1 = 0.3
    m2 = 0.6
    if var1 == 'M300' and var2 == 'Mtot':
        e_t_band = [subhalo_e_t_1.int_over_density(m1) / subhalo_e_t_1.int_over_density(subhalo_e_t_1.max_radius),
                    subhalo_e_t_2.int_over_density(m1) / subhalo_e_t_2.int_over_density(subhalo_e_t_2.max_radius)]
        n_t_band = [subhalo_n_t_1.int_over_density(m1) / subhalo_n_t_1.int_over_density(subhalo_n_t_1.max_radius),
                    subhalo_n_t_2.int_over_density(m1) / subhalo_n_t_2.int_over_density(subhalo_n_t_2.max_radius)]
        e_nt_band = [subhalo_e_nt_1.int_over_density(m1) / subhalo_e_nt_1.int_over_density(subhalo_e_nt_1.max_radius),
                    subhalo_e_nt_2.int_over_density(m1) / subhalo_e_nt_2.int_over_density(subhalo_e_nt_2.max_radius)]
        n_nt_band = [subhalo_n_nt_1.int_over_density(m1) / subhalo_n_nt_1.int_over_density(subhalo_n_nt_1.max_radius),
                    subhalo_n_nt_2.int_over_density(m1) / subhalo_n_nt_2.int_over_density(subhalo_n_nt_2.max_radius)]
        e_bert_band = [subhalo_e_bert_1.int_over_density(m1) / subhalo_e_bert_1.int_over_density(subhalo_e_bert_1.max_radius),
                    subhalo_e_bert_2.int_over_density(m1) / subhalo_e_bert_2.int_over_density(subhalo_e_bert_2.max_radius)]
        n_bert_band = [subhalo_n_bert_1.int_over_density(m1) / subhalo_n_bert_1.int_over_density(subhalo_n_bert_1.max_radius),
                    subhalo_n_bert_2.int_over_density(m1) / subhalo_n_bert_2.int_over_density(subhalo_e_bert_2.max_radius)]
        m1labl = 'M300'
        m2labl = 'Mtot'
    elif var1 == 'M300' and var2 == 'M600':
        e_t_band = [subhalo_e_t_1.int_over_density(m1) / subhalo_e_t_1.int_over_density(m2),
                    subhalo_e_t_2.int_over_density(m1) / subhalo_e_t_2.int_over_density(m2)]
        n_t_band = [subhalo_n_t_1.int_over_density(m1) / subhalo_n_t_1.int_over_density(m2),
                    subhalo_n_t_2.int_over_density(m1) / subhalo_n_t_2.int_over_density(m2)]
        e_nt_band = [subhalo_e_nt_1.int_over_density(m1) / subhalo_e_nt_1.int_over_density(m2),
                     subhalo_e_nt_2.int_over_density(m1) / subhalo_e_nt_2.int_over_density(m2)]
        n_nt_band = [subhalo_n_nt_1.int_over_density(m1) / subhalo_n_nt_1.int_over_density(m2),
                     subhalo_n_nt_2.int_over_density(m1) / subhalo_n_nt_2.int_over_density(m2)]
        e_bert_band = [subhalo_e_bert_1.int_over_density(m1) / subhalo_e_bert_1.int_over_density(m2),
                        subhalo_e_bert_2.int_over_density(m1) / subhalo_e_bert_2.int_over_density(m2)]
        n_bert_band = [subhalo_n_bert_1.int_over_density(m1) / subhalo_n_bert_1.int_over_density(m2),
                        subhalo_n_bert_2.int_over_density(m1) / subhalo_n_bert_2.int_over_density(m2)]
        m1labl = 'M300'
        m2labl = 'M600'
    elif var1 == 'Mrmax' and var2 == 'Mtot':
        rmax1, vmax1 = rmax_vmax(np.log10(min_mass))
        rmax2, vmax2 = rmax_vmax(np.log10(max_mass))

        e_t_band = [subhalo_e_t_1.int_over_density(rmax1) / subhalo_e_t_1.int_over_density(subhalo_e_t_1.max_radius),
                    subhalo_e_t_2.int_over_density(rmax2) / subhalo_e_t_2.int_over_density(subhalo_e_t_2.max_radius)]
        n_t_band = [subhalo_n_t_1.int_over_density(rmax1) / subhalo_n_t_1.int_over_density(subhalo_n_t_1.max_radius),
                    subhalo_n_t_2.int_over_density(rmax2) / subhalo_n_t_2.int_over_density(subhalo_n_t_2.max_radius)]
        e_nt_band = [subhalo_e_nt_1.int_over_density(rmax1) / subhalo_e_nt_1.int_over_density(subhalo_e_nt_1.max_radius),
                     subhalo_e_nt_2.int_over_density(rmax2) / subhalo_e_nt_2.int_over_density(subhalo_e_nt_2.max_radius)]
        n_nt_band = [subhalo_n_nt_1.int_over_density(rmax1) / subhalo_n_nt_1.int_over_density(subhalo_n_nt_1.max_radius),
                     subhalo_n_nt_2.int_over_density(rmax2) / subhalo_n_nt_2.int_over_density(subhalo_n_nt_2.max_radius)]
        e_bert_band = [subhalo_e_bert_1.int_over_density(rmax1) / subhalo_e_bert_1.int_over_density(subhalo_e_bert_1.max_radius),
                       subhalo_e_bert_2.int_over_density(rmax2) / subhalo_e_bert_2.int_over_density(subhalo_e_bert_2.max_radius)]
        n_bert_band = [subhalo_n_bert_1.int_over_density(rmax1) / subhalo_n_bert_1.int_over_density(subhalo_n_bert_1.max_radius),
                        subhalo_n_bert_2.int_over_density(rmax2) / subhalo_n_bert_2.int_over_density(subhalo_e_bert_2.max_radius)]
        m1labl = 'Mrmax'
        m2labl = 'Mtot'


    ax.axhspan(e_t_band[0], e_t_band[1], alpha=0.2, color=color_list[0])
#    ax.axhspan(n_t_band[0], n_t_band[1], alpha=0.1, color=color_list[1])
    ax.axhspan(e_nt_band[0], e_nt_band[1], alpha=0.1, color=color_list[2])
#    ax.axhspan(n_nt_band[0], n_nt_band[1], alpha=0.1, color=color_list[3])
    ax.axhspan(e_bert_band[0], e_bert_band[1], alpha=0.1, color=color_list[4])
#    ax.axhspan(n_bert_band[0], e_bert_band[1], alpha=0.1, color=color_list[5])

    plt.text(1.5, .1,
             r'Halo Masses: [{:.1e}, {:.1e}] $M_\odot$'.format(min_mass, max_mass),
             fontsize=10, ha='left', va='center', color='Black')
    plt.xlabel('Distance from GC  [kpc]', fontsize=18.)
    plt.ylabel(m1labl + '/' + m2labl, fontsize=18.)
    fig.set_tight_layout(True)
    fname = 'Mass_Ratio_Analysis_mlow_{:.2e}'.format(min_mass) + '_mhigh_{:.2e}'.format(max_mass) +\
        '_' + m1labl + '_to_' + m2labl + '.pdf'
    pl.savefig(dir + '/Via_Lac_plots/' + fname)
    return


def number_subhalo_histogram(mass_low=10. ** 4, mass_high = 10.**8., gcd_low=0.,
                             gcd_high=4000., nbins=20):

    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ViaLacteaII_Info.dat'
    via_lac_file = np.loadtxt(dir + f_name)
    m_bins = np.logspace(np.log10(mass_low), np.log10(mass_high), nbins)

    sub_of_int = Via_Lactea_II().find_subhalos(min_mass=4.1 * 10 ** 3., max_mass=10. ** 12.,
                                               gcd_min=gcd_low, gcd_max=gcd_high, print_info=False)
    ntot = via_lac_file[sub_of_int][:, 5].size
    plt.hist(via_lac_file[sub_of_int][:, 5], bins=m_bins, log=True, normed=True,
             weights=np.ones(ntot), alpha=0.3, color='Blue', label=r'$\frac{N}{\Delta M}$')
    plt.hist(via_lac_file[sub_of_int][:, 5], bins=m_bins, log=True, normed=True,
             weights=via_lac_file[sub_of_int][:, 5] * ntot, alpha=0.3, color='Red', label=r'$\frac{M \times N}{\Delta M}$')
    pl.xlim([mass_low, mass_high])
    pl.gca().set_xscale("log")
    plt.xlabel(r'Mass  [$M_\odot$]', fontsize=16)
    plt.legend()
    plt.text(10.**7, 10.**-6.,
             'GC Dist: [{:.1f}, {:.1f}] kpc'.format(gcd_low, gcd_high),
             fontsize=10, ha='left', va='center', color='Black')
    fname = 'Number_Histogram_mlow_{:.2e}'.format(mass_low) + '_mhigh_{:.2e}'.format(mass_high) + \
            '_GCDlow_{:.1f}'.format(gcd_low) + '_GCDhigh_{:.1f}'.format(gcd_high) + '.pdf'
    pl.savefig(dir + '/Via_Lac_plots/' + fname)

    return
