import matplotlib as mpl
try:
    mpl.use('Agg')
except:
    pass
import numpy as np
from subhalo import *
import os
import pickle
from scipy.interpolate import interp1d, interp2d,RectBivariateSpline
from scipy.optimize import fminbound
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import itertools
from profiles import *
from ELVIS_Analysis import *
from Via_LacteaII_Analysis import *

#  mpl.use('pdf')
rc('font', **{'family': 'serif', 'serif': ['Times', 'Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16


def profile_comparison(plot_hw=True):
    minrad = -3.
    mass_list = [10. ** 4., 10 ** 5., 10 ** 6., 10 ** 7.]
    rb_list = np.logspace(-4, np.log10(1.), 20)
    gamma_list = np.linspace(0.2, 1.45, 20)
    fig = plt.figure()
    f, ax = plt.subplots(2, 2, sharex='col', sharey='row')

    ii = 0
    for i, m_sub in enumerate(mass_list):
        if i == 2:
            ii = 1
        e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
        n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True)
        r_hw = np.logspace(minrad, np.log10(HW_Fit(m_sub).max_radius), 100)
        den_hw = np.zeros(rb_list.size * gamma_list.size * r_hw.size).reshape((rb_list.size,
                                                                               gamma_list.size, r_hw.size))
        for j, rb in enumerate(rb_list):
            for k, gam in enumerate(gamma_list):
                for z, r in enumerate(r_hw):
                    den_hw[j, k, z] = HW_Fit(m_sub, gam=gam, rb=rb).density(r) * \
                                      hw_prob_rb(rb, m_sub) * hw_prob_gamma(gam) * \
                                      GeVtoSolarM * kpctocm ** 3.
        hw_marg = np.zeros_like(r_hw)
        for j in range(len(r_hw)):
            hw_marg[j] = RectBivariateSpline(rb_list, gamma_list,
                                             den_hw[:, :, j]).integral(np.min(rb_list), np.max(rb_list),
                                                                       np.min(gamma_list), np.max(gamma_list))


        r1_e_tr = np.logspace(minrad, np.log10(e_tr.max_radius), 100)
        r_n_ntr = np.logspace(minrad, np.log10(n_ntr.max_radius), 100)

        den_e_tr1 = e_tr.density(r1_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr = n_ntr.density(r_n_ntr) * GeVtoSolarM * kpctocm ** 3.

        ax[i%2, ii].set_xscale("log")
        ax[i%2, ii].set_yscale('log')
        ax[i%2, ii].set_xlim([10 ** minrad, 20.])
        ax[i%2, ii].plot(r1_e_tr, den_e_tr1, lw=2, color='Black')
        #ax[i%2, ii].plot(r2_e_tr, den_e_tr2, '--', lw=2, ms=1, color='Black')
        ax[i%2, ii].plot(r_n_ntr, den_n_ntr, lw=2, color='Magenta')
        #ax[i%2, ii].plot(r2_n_ntr, den_n_ntr2, '--', ms=1, lw=2, color='Magenta')
        ax[i%2, ii].plot(r_hw, hw_marg, lw=2, color='Blue')
        ax[i % 2, ii].axvline(x=r1_e_tr[-1], ymin=0,
                              ymax=(np.log10(den_e_tr1[-1]) - np.log10(2 * 10 ** 4.)) /
                                   (11. - np.log10(2. * 10 ** 4.)),
                              linewidth=2, color='Black')
        m1, m2 = sci_note('{:.1e}'.format(m_sub))

        ax[i % 2, ii].text(9, 10**10, r'Subhalo Mass: ${} \times 10^{} M_\odot$'.format(m1, m2),
                           fontsize=12, ha='right')
        ax[i%2, ii].set_yticklabels(['', r'$10^4$', '', r'$10^6$', '', r'$10^8$', '', r'$10^{10}$', ''])

        if i % 2 == 1:
            ax[i % 2, ii].set_xlabel('Radius [kpc]', fontsize=16)
            if i == 1:
                ax[i % 2, ii].set_xticklabels(['', '', r'$10^{-2}$', r'$10^{-1}$', '1', r'$10$'])
            else:
                pass
                ax[i % 2, ii].set_xticklabels(['', '', r'$10^{-2}$', r'$10^{-1}$', '1', r'$10$'])
        if ii == 0:
            ax[i%2, ii].set_ylabel(r'$\rho$ ' + r'[$M_\odot$ / kpc^3]', fontsize=16)
            if i == 0:
                ax[i%2, ii].text(.4, 10**8.5, 'Bertoni et al.', fontsize=10, color='k')
                ax[i%2, ii].text(.4, 10 ** 7.8, 'Schoonenberg et al.', fontsize=10, color='Magenta')
                ax[i%2, ii].text(.4, 10 ** 7., 'This Work', fontsize=10, color='blue')
        ax[i%2, ii].set_ylim([2. * 10. ** 4., 10. ** 11.])
    #pl.suptitle('Profile Comparison', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Multi_plot_Density_Comparison_msubhalo_' + \
        '_BLH_Bertone_HW.pdf'
    plt.subplots_adjust(wspace=0.0, hspace=0.)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def subhalo_comparison():
    rb_list = np.logspace(-3, np.log10(1.), 20)
    gamma_list = np.linspace(0.2, 0.85 + 0.351 / 0.861 - 0.1, 20)
    via_lac = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ViaLacteaII_Useable_Subhalos.dat')
    elvis = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ELVIS_Useable_Subhalos.dat')
    vl_num = [5266, 5216]
    elv_num = [3444, 20801]
    sub_list = []
    for i in vl_num:
        sub_list.append(via_lac[i])
        print 'Via Lactea:'
        print via_lac[i]
    for i in elv_num:
        sub_list.append(elvis[i])
        print 'ELVIS:'
        print elvis[i]


    minrad = -2.

    fig = plt.figure()
    f, ax = plt.subplots(2, 2, sharex='col', sharey='row')
    ii = 0

    for i, sub in enumerate(sub_list):
        if i == 2:
            ii = 1
        m_sub = sub[5]
        vmax = sub[3]
        rmax = sub[4]
        m_at_rmax = rmax * vmax ** 2. / newton_G
        try:
            gam = find_gamma(m_sub, vmax, rmax, error=0.)
        except ValueError:
            gam = 0.
        e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
        n_ntr = NFW(m_sub, 1., rmax=rmax, vmax=vmax, arxiv_num=160106781)
        hw = KMMDSM(m_sub, gam, rmax=rmax, vmax=vmax)
        n_ntr2 = NFW(m_sub, 1., arxiv_num=160304057)
        hw2 = HW_Fit(m_sub)

        r_hw = np.logspace(minrad, np.log10(10.), 100)
        m_inr_e = np.zeros(100)
        m_inr_n = np.zeros(100)
        m_inr_n2 = np.zeros(100)
        m_inr_h = np.zeros(100)
        m_inr_h2 = np.zeros(100)
        for j in range(100):
            m_inr_e[j] = e_tr.int_over_density(r_hw[j])
            m_inr_n[j] = n_ntr.int_over_density(r_hw[j])
            m_inr_n2[j] = n_ntr2.int_over_density(r_hw[j])
            m_inr_h[j] = hw.int_over_density(r_hw[j])
            m_inr_h2[j] = hw2.int_over_density(r_hw[j])

        m_hw = np.zeros(rb_list.size *
                        gamma_list.size *
                        r_hw.size).reshape((rb_list.size,
                                            gamma_list.size, r_hw.size))
        for j, rb in enumerate(rb_list):
            for k, gam in enumerate(gamma_list):
                for z, r in enumerate(r_hw):
                    m_hw[j, k, z] = HW_Fit(m_sub, gam=gam, rb=rb).int_over_density(r) * \
                                      hw_prob_rb(rb, m_sub) * hw_prob_gamma(gam)
        hw_marg = np.zeros_like(r_hw)
        for j in range(len(r_hw)):
            hw_marg[j] = RectBivariateSpline(rb_list, gamma_list,
                                             m_hw[:, :, j]).integral(np.min(rb_list), np.max(rb_list),
                                                                       np.min(gamma_list), np.max(gamma_list))

        ax[i%2, ii].set_xscale("log")
        ax[i%2, ii].set_yscale('log')
        ax[i%2, ii].set_xlim([10 ** minrad, 1. * 10. ** 1.])
        ax[i%2, ii].plot(r_hw, m_inr_e, lw=2, color='Black')
        ax[i%2, ii].plot(r_hw, m_inr_n, lw=2, color='Magenta')
        #ax[i % 2, ii].plot(r_hw, m_inr_n2, lw=1, color='Magenta')
        ax[i%2, ii].plot(r_hw, m_inr_h, lw=2, color='Blue')
        #ax[i % 2, ii].plot(r_hw, m_inr_h2, lw=1, color='Blue')
        #ax[i % 2, ii].plot(r_hw, hw_marg, lw=1, color='Blue')
        m1, m2 = sci_note('{:.1e}'.format(m_sub))
        ax[i%2, ii].text(7., 2*10**4, r'Subhalo Mass: ${} \times 10^{} M_\odot$'.format(m1,m2),
                         fontsize=10, ha='right',va='bottom')
        ax[i%2, ii].set_yticklabels(['', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$', '', ''])
        ax[i%2, ii].plot(rmax, m_at_rmax, '*', ms=10, mec='Red', mfc='Red')
        ax[i % 2, ii].plot(9., m_sub, '*', ms=10, mec='Red', mfc='Red')
        if i % 2 == 1:
            ax[i % 2, ii].set_xlabel('Radius [kpc]', fontsize=16)
            if i == 1:
                ax[i % 2, ii].set_xticklabels(['', r'$10^{-2}$', r'$10^{-1}$', '1', ''])
            else:
                ax[i % 2, ii].set_xticklabels(['', '', r'$10^{-1}$', '1', r'$10$'])
        if ii == 0:
            ax[i%2, ii].set_ylabel(r'$M(r)$   $[M_\odot]$', fontsize=16)
            if i == 0:
                ax[i%2, ii].text(2 * 10**-2., 10**7.6, 'Bertoni et al.', fontsize=10, color='k')
                ax[i%2, ii].text(2 * 10**-2., 10**7.3, 'Schoonenberg et al.', fontsize=10, color='Magenta')
                ax[i%2, ii].text(2 * 10**-2., 10**7., 'This Work', fontsize=10, color='blue')
        ax[i%2, ii].set_ylim([10. ** 4., 10. ** 8.])
    #pl.suptitle('Profile Comparison', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/PaperPlots/' + 'Sim_Comparison_msubhalo_BLH_Bertone_HW.pdf'
    plt.subplots_adjust(wspace=0.0, hspace=0.)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def Jfactor_plots(m_sub=10.**7., dist=1.):
    rb_list = np.logspace(-5, np.log10(1.), 30)
    gamma_list = np.linspace(0., 1.45, 20)
    # J vs Dist
    e_tr = Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
    n_ntr = NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True)
    hw_fit = HW_Fit(m_sub, M200=True)

    num_dist_pts = 30
    dist_tab = np.logspace(-3., 1., num_dist_pts)
    e_tr_j = np.power(10, e_tr.J_pointlike(dist_tab))
    n_ntr_j = np.power(10, n_ntr.J_pointlike(dist_tab))

    m_hw = np.zeros(rb_list.size *
                    gamma_list.size *
                    num_dist_pts).reshape((rb_list.size,
                                        gamma_list.size, num_dist_pts))
    for j, rb in enumerate(rb_list):
        for k, gam in enumerate(gamma_list):
            for z, r in enumerate(dist_tab):
                a = HW_Fit(m_sub, gam=gam, rb=rb)
                m_hw[j, k, z] = np.power(10, a.J_pointlike(r)) * \
                                hw_prob_rb(rb, m_sub) * hw_prob_gamma(gam)
    hw_marg = np.zeros_like(dist_tab)
    for j in range(num_dist_pts):
        hw_marg[j] = RectBivariateSpline(rb_list, gamma_list,
                                         m_hw[:, :, j]).integral(np.min(rb_list), np.max(rb_list),
                                                                 np.min(gamma_list), np.max(gamma_list))

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    pl.xlim([10 ** -3., 10. ** 1.])
    pl.ylim([10 ** 20., np.max([e_tr_j[0], hw_marg[0]])])
    plt.plot(dist_tab, e_tr_j, lw=1, color='Black', label='BLH')
    plt.plot(dist_tab, n_ntr_j, lw=1, color='Magenta', label='Bertone')
    plt.plot(dist_tab, hw_marg, lw=1, color='Blue', label='HW')

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
    test = np.zeros(mass_tab.size)

    for i, m_sub in enumerate(mass_tab):
        #print i + 1, '/', mass_tab.size
        e_tr_j[i] = 10. ** Einasto(m_sub / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False).J_pointlike(dist)
        n_ntr_j[i] = 10. ** NFW(m_sub, .16, truncate=False, arxiv_num=160106781, M200=True).J_pointlike(dist)
        test[i] = 10. ** HW_Fit(m_sub).J_pointlike(dist)

    m_hw = np.zeros(rb_list.size *
                    gamma_list.size *
                    mass_tab.size).reshape((rb_list.size,
                                           gamma_list.size, mass_tab.size))
    for j, rb in enumerate(rb_list):
        for k, gam in enumerate(gamma_list):
            for z, m in enumerate(mass_tab):
                a = HW_Fit(m, gam=gam, rb=rb)
                m_hw[j, k, z] = np.power(10, a.J_pointlike(dist)) * \
                                hw_prob_rb(rb, m) * hw_prob_gamma(gam)
    hw_marg = np.zeros_like(mass_tab)
    for j in range(mass_tab.size):
        hw_marg[j] = RectBivariateSpline(rb_list, gamma_list,
                                         m_hw[:, :, j]).integral(np.min(rb_list), np.max(rb_list),
                                                                 np.min(gamma_list), np.max(gamma_list))

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    pl.xlim([10 ** 0., 10. ** 7.])
    pl.ylim([np.min([n_ntr_j]), np.max([e_tr_j])])
    plt.plot(mass_tab, e_tr_j, lw=1, color='Black', label='BLH')
    plt.plot(mass_tab, n_ntr_j, lw=1, color='Magenta', label='Bertone')
    plt.plot(mass_tab, hw_marg, lw=1, color='Blue', label='HW')
    plt.plot(mass_tab, test, lw=1, color='green', label='HW')


    pl.xlabel(r'Mass [$M_\odot$]', fontsize=20)
    pl.ylabel(r'J   [$GeV^2 / cm^5$]', fontsize=20)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Jfactor_vs_Mass_Distance_{:.2f}'.format(dist) + \
               '_BLH_Bertone_HW.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)

    return


def fractional_extension(mx=10., cs=2.2*10**-26., annih='BB', calc=False):

    thresh_tab = np.logspace(-10., -9., 5)
    mass_tab = np.logspace(5., 7., 5)
    gam_tab = np.linspace(.1, 1.3, 7)

    thres_p = np.zeros(5)
    thres_p1 = np.zeros(5)
    thres_p3 = np.zeros(5)
    thres_pp = np.zeros(5)
    thres_pp1 = np.zeros(5)
    thres_pp3 = np.zeros(5)

    count = 0

    if calc:
        for i, th in enumerate(thresh_tab):
            print 'Threshold: {:.2e}'.format(th)
            m_p = np.zeros_like(mass_tab)
            m_p1 = np.zeros_like(mass_tab)
            m_p3 = np.zeros_like(mass_tab)
            for o,mass in enumerate(mass_tab):
                print '     Mass: {:.2e}'.format(mass)
                rb_med = np.log10(10. ** (-3.945) * mass ** 0.421)
                rb_low = rb_med - .5
                rb_high = rb_med + .5
                rb_list = np.logspace(rb_low, rb_high, 6)
                rb_p = np.zeros_like(rb_list)
                rb_p1 = np.zeros_like(rb_list)
                rb_p3 = np.zeros_like(rb_list)
                for z, rb in enumerate(rb_list):
                    print 'Threshold: {:.2e}    Mass: {:.2e}     Rb: {:.2e}'.format(th,mass,rb)
                    gam_p = np.zeros_like(gam_tab)
                    gam_p1 = np.zeros_like(gam_tab)
                    gam_p3 = np.zeros_like(gam_tab)
                    for k, gam in enumerate(gam_tab):
                        print 'Threshold: {:.2e}    Mass: {:.2e}     Rb: {:.2e}    Gamma: {:.2f}'.format(th, mass, rb,gam)
                        prof = HW_Fit(mass, gam=gam, rb=rb)
                        mod = Model(mx, cs, annih, mass, profile=2., m200=True, gam=gam, rb=rb)

                        gam_p[k] = mod.d_max_point(th)
                        dis_tab = np.logspace(-1., np.log10(gam_p[k]), 7)
                        sig68 = np.zeros_like(dis_tab)
                        for j, d in enumerate(dis_tab):
                            try:
                                print 'Distance: ', d
                                if 10. ** (prof.J(d, 0.01) / prof.J_pointlike(d)) > 0.68:
                                    raise ValueError
                                sig68[j] = prof.Spatial_Extension(d, thresh_calc=False)
                            except:
                                pass
                        #print sig68
                        dis_tab = dis_tab[(sig68 < 85.) & (sig68 > 0.01)]
                        sig68 = sig68[(sig68 < 85.) & (sig68 > 0.01)]
                        extension = interp1d(np.log10(dis_tab), np.log10(sig68), kind='linear',
                                             fill_value='extrapolate', bounds_error=False)
                        gam_p1[k] = np.power(10, fminbound(lambda x: np.abs(10. ** extension(x) - 0.1), -3.,
                                                           np.log10(gam_p[k])))
                        gam_p3[k] = np.power(10, fminbound(lambda x: np.abs(10. ** extension(x) - 0.3), -3.,
                                                           np.log10(gam_p[k])))

                        count += 1
                    gamma_full = np.linspace(.01, 1.45, 100)
                    print 'Gamma Tabs'
                    print gam_p
                    print gam_p1
                    print gam_p3
                    gamma_interp_p = 10. ** interpola(np.log10(gamma_full), np.log10(gam_tab), np.log10(gam_p)) * \
                                      hw_prob_gamma(gamma_full)
                    gamma_interp_p1 = 10. ** interpola(np.log10(gamma_full), np.log10(gam_tab), np.log10(gam_p1)) * \
                                      hw_prob_gamma(gamma_full)
                    gamma_interp_p3 = 10. ** interpola(np.log10(gamma_full), np.log10(gam_tab), np.log10(gam_p3)) * \
                                      hw_prob_gamma(gamma_full)

                    rb_p[z] = np.trapz(gamma_interp_p, gamma_full)
                    rb_p1[z] = np.trapz(gamma_interp_p1, gamma_full)
                    rb_p3[z] = np.trapz(gamma_interp_p3, gamma_full)

                rb_full = np.logspace(rb_low, rb_high, 100)
                rb_interp_p = 10. ** interp1d(np.log10(rb_list), np.log10(rb_p), kind='linear',
                                               bounds_error=False, fill_value='extrapolate')(np.log10(rb_full))
                rb_interp_p1 = 10. ** interp1d(np.log10(rb_list), np.log10(rb_p1), kind='linear',
                                                  bounds_error=False, fill_value='extrapolate')(np.log10(rb_full))
                rb_interp_p3 = 10. ** interp1d(np.log10(rb_list), np.log10(rb_p3), kind='linear',
                                                  bounds_error=False, fill_value='extrapolate')(np.log10(rb_full))
                m_p[o] = np.trapz(rb_interp_p * hw_prob_rb(rb_full, mass), rb_full)
                m_p1[o] = np.trapz(rb_interp_p1 * hw_prob_rb(rb_full, mass), rb_full)
                m_p3[o] = np.trapz(rb_interp_p3 * hw_prob_rb(rb_full, mass), rb_full)
            print 'Mass Tabs: '
            print m_p
            print m_p1
            print m_p3
            m_all = np.logspace(5, 7, 100)
            m_interp_p = 10. ** interp1d(np.log10(mass_tab), np.log10(m_p), kind='linear',
                                          bounds_error=False, fill_value='extrapolate')(np.log10(m_all))
            m_interp_p1 = 10. ** interp1d(np.log10(mass_tab), np.log10(m_p1), kind='linear',
                                           bounds_error=False, fill_value='extrapolate')(np.log10(m_all))
            m_interp_p3 = 10. ** interp1d(np.log10(mass_tab), np.log10(m_p3), kind='linear',
                                           bounds_error=False, fill_value='extrapolate')(np.log10(m_all))

            thres_p[i] = np.trapz(m_interp_p ** 3. * m_all ** (-1.9), m_all)
            thres_p1[i] = np.trapz(m_interp_p1 ** 3. * m_all ** (-1.9), m_all)
            thres_p3[i] = np.trapz(m_interp_p3 ** 3. * m_all ** (-1.9), m_all)
    else:
        file = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Fractional_Ext.dat')
        thresh_tab = np.unique(file[:, 0])
        for i in range(thresh_tab.size):
            print 'Mass Tabs 100 GeV: '
            m_p = file[file[:, 1] == 0.0][i][2:]
            m_p1 = file[file[:, 1] == 0.1][i][2:]
            m_p3 = file[file[:, 1] == 0.3][i][2:]
            # print m_p
            # print m_p1
            # print m_p3
            m_all = np.logspace(5, 7, 100)
            m_interp_p = 10. ** interp1d(np.log10(mass_tab), np.log10(m_p), kind='linear',
                                         bounds_error=False, fill_value='extrapolate')(np.log10(m_all))
            m_interp_p1 = 10. ** interp1d(np.log10(mass_tab), np.log10(m_p1), kind='linear',
                                          bounds_error=False, fill_value='extrapolate')(np.log10(m_all))
            m_interp_p3 = 10. ** interp1d(np.log10(mass_tab), np.log10(m_p3), kind='linear',
                                          bounds_error=False, fill_value='extrapolate')(np.log10(m_all))

            thres_p[i] = np.trapz(m_interp_p ** 3. * m_all ** (-1.9), m_all)
            thres_p1[i] = np.trapz(m_interp_p1 ** 3. * m_all ** (-1.9), m_all)
            thres_p3[i] = np.trapz(m_interp_p3 ** 3. * m_all ** (-1.9), m_all)

        file2 = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Fractional_Ext_10GeV.dat')
        #thresh_tab = np.unique(file[:, 0])
        for i in range(thresh_tab.size):
            print 'Mass Tabs 10 GeV: '
            m_pp = file2[file2[:, 1] == 0.0][i][2:]
            m_pp1 = file2[file2[:, 1] == 0.1][i][2:]
            m_pp3 = file2[file2[:, 1] == 0.3][i][2:]
            # print m_pp
            # print m_pp1
            # print m_pp3
            #m_all = np.logspace(5, 7, 100)
            m_interp_pp = 10. ** interp1d(np.log10(mass_tab), np.log10(m_pp), kind='linear',
                                         bounds_error=False, fill_value='extrapolate')(np.log10(m_all))
            m_interp_pp1 = 10. ** interp1d(np.log10(mass_tab), np.log10(m_pp1), kind='linear',
                                          bounds_error=False, fill_value='extrapolate')(np.log10(m_all))
            m_interp_pp3 = 10. ** interp1d(np.log10(mass_tab), np.log10(m_pp3), kind='linear',
                                          bounds_error=False, fill_value='extrapolate')(np.log10(m_all))

            thres_pp[i] = np.trapz(m_interp_pp ** 3. * m_all ** (-1.9), m_all)
            thres_pp1[i] = np.trapz(m_interp_pp1 ** 3. * m_all ** (-1.9), m_all)
            thres_pp3[i] = np.trapz(m_interp_pp3 ** 3. * m_all ** (-1.9), m_all)

    #print 'Threshold Tabs: '
    #print thres_p
    #print thres_p1
    #print thres_p3
    print 'Mass Comparison: ', np.column_stack((thres_p3 / thres_p, thres_pp3 / thres_pp))
    thresh_full = np.logspace(-10., -9., 40)
    plt_ext1 = 10. ** interpola(np.log10(thresh_full), np.log10(thresh_tab), np.log10(thres_p1 / thres_p))
    plt_ext3 = 10. ** interpola(np.log10(thresh_full), np.log10(thresh_tab), np.log10(thres_p3 / thres_p))
    plt_ext3_10 = 10. ** interpola(np.log10(thresh_full), np.log10(thresh_tab), np.log10(thres_pp3 / thres_pp))
    plt_ext1_10 = 10. ** interpola(np.log10(thresh_full), np.log10(thresh_tab), np.log10(thres_pp1 / thres_pp))
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    pl.xlim([10 ** -10., 10. ** -9.])
    pl.ylim([10**-2., .5])
    plt.plot(thresh_full, plt_ext1, lw=2, color='Purple')
    plt.plot(thresh_full, plt_ext3, lw=2, color='goldenrod')
    plt.plot(thresh_full, plt_ext3_10, '--', lw=2, color='goldenrod')
    plt.plot(thresh_full, plt_ext1_10, '--', lw=2, color='purple')
    pl.xlabel(r'$\Phi_{{min}}$ [$cm^{{-2}}s^{{-1}}$]', fontsize=20)
    pl.ylabel(r'$N_{{obs}}(\sigma_{{68}} > \sigma_{{min}}) / N_{{obs}}$', fontsize=20)
    pl.text(9.*10**-10., 1.4*10**-2, r'$\sigma_{{min}} = 0.1^{{\circ}}$', fontsize=14, color='purple',
            verticalalignment='bottom', horizontalalignment='right')
    pl.text(9. * 10 ** -10., 1.1*10**-2, r'$\sigma_{{min}} = 0.3^{{\circ}}$', fontsize=14, color='goldenrod',
            verticalalignment='bottom', horizontalalignment='right')

    pl.text(3. * 10 ** -10., 5.5 * 10 ** -2, r'100 GeV', fontsize=14, color='k',
            verticalalignment='bottom', horizontalalignment='right', rotation = 20)
    pl.text(3. * 10 ** -10., 1.3 * 10 ** -2, r'10 GeV', fontsize=14, color='k',
            verticalalignment='bottom', horizontalalignment='right', rotation=22)
    pl.text(1.4 * 10 ** -10., .15, r'100 GeV', fontsize=14, color='k',
            verticalalignment='bottom', horizontalalignment='right', rotation=16)
    pl.text(1.4 * 10 ** -10., .0515, r'10 GeV', fontsize=14, color='k',
            verticalalignment='bottom', horizontalalignment='right', rotation=18)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Fractional_w_SpatialExtension.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def hw_prob_rb(rb, mass):
    rb_norm = 10. ** (-3.945) * mass ** 0.421
    sigma_c = 0.496
    return (np.exp(- (np.log(rb / rb_norm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
            (np.sqrt(2. * np.pi) * sigma_c * rb))


def hw_prob_gamma(gam):
    # norm inserted b/c integration truncated on region [0, 1.45]
    sigma = 0.42
    k = 0.1
    mu = 0.74
    y = -1. / k * np.log(1. - k * (gam - mu) / sigma)
    norm = quad(lambda x: np.exp(- -1. / k * np.log(1. - k * (x - mu) / sigma) ** 2. / 2.)
                          / (np.sqrt(2. * np.pi) * (sigma - k * (x - mu))), 0., 1.45)[0]
    return np.exp(- y ** 2. / 2.) / (np.sqrt(2. * np.pi) * (sigma - k * (gam - mu))) / norm

def sigma_68(mx=100., cs=2.2*10**-26., annih='BB'):
    hw7 = HW_Fit(10**7, gam=0.74)
    model_hw7 = Model(mx, cs, annih, 10**7, profile=2, m200=True, rb=hw7.rb)

    hw6 = HW_Fit(10**6, gam=0.74)
    model_hw6 = Model(mx, cs, annih, 10**6, profile=2, m200=True, rb=hw6.rb)

    hw5 = HW_Fit(10**5, gam=0.74)
    model_hw5 = Model(mx, cs, annih, 10**5, profile=2, m200=True, rb=hw5.rb)

    d_list = np.logspace(-1., 1., 10)
    sig68_7 = np.zeros(d_list.size * 2).reshape((d_list.size, 2))
    sig68_6 = np.zeros(d_list.size * 2).reshape((d_list.size, 2))
    sig68_5 = np.zeros(d_list.size * 2).reshape((d_list.size, 2))
    d_list_full = np.logspace(-2., 1., 150)
    for i, d in enumerate(d_list):
        print i+1, '/', len(d_list)
        val = hw7.Spatial_Extension(d)
        if 0.1 < val < 2.0:
            sig68_7[i] = [d, val]
        val = hw6.Spatial_Extension(d)
        if 0.1 < val < 2.0:
            sig68_6[i] = [d, val]
        val = hw5.Spatial_Extension(d)
        if 0.1 < val < 2.0:
            sig68_5[i] = [d, val]

    print 'Obtaining Dmax Extended'
    dmax_7 = model_hw7.D_max_extend()
    dmax_6 = model_hw6.D_max_extend()
    dmax_5 = model_hw5.D_max_extend()
    print dmax_7, dmax_6, dmax_5

    print 'Plotting'
    sig68_7 = sig68_7[sig68_7[:, 1] > 0.]
    sig68_6 = sig68_6[sig68_6[:, 1] > 0.]
    sig68_5 = sig68_5[sig68_5[:, 1] > 0.]
    s7_plot = np.polyfit(np.log10(sig68_7[:, 0]), np.log10(sig68_7[:, 1]), 1)
    s6_plot = np.polyfit(np.log10(sig68_6[:, 0]), np.log10(sig68_6[:, 1]), 1)
    s5_plot = np.polyfit(np.log10(sig68_5[:, 0]), np.log10(sig68_5[:, 1]), 1)
    s7_p = 10. ** np.polyval(s7_plot, np.log10(d_list_full))
    s6_p = 10. ** np.polyval(s6_plot, np.log10(d_list_full))
    s5_p = 10. ** np.polyval(s5_plot, np.log10(d_list_full))

    dots7 = 10 ** np.polyval(s7_plot, np.log10(dmax_7))
    dots6 = 10 ** np.polyval(s6_plot, np.log10(dmax_6))
    dots5 = 10 ** np.polyval(s5_plot, np.log10(dmax_5))

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')

    xmin = sig68_5[0, 0]
    pl.xlim([xmin, 10. ** 1.])
    pl.ylim([0.1, 1.9])
    pl.xlabel(r'Distance   [kpc]', fontsize=20)
    pl.ylabel(r'$\sigma_{{68}}$   [$\deg$]', fontsize=20)
    pl.plot(d_list_full, s7_p, lw=2, color='k')
    pl.plot(d_list_full, s6_p, lw=2, color='blueviolet')
    pl.plot(d_list_full, s5_p, lw=2, color='blue')
    pl.plot(dmax_7, dots7, 'o', ms=16, mfc='k', mec='k')
    pl.plot(dmax_6, dots6, 'o', ms=16, mfc='blueviolet', mec='blueviolet')
    pl.plot(dmax_5, dots5, 'o', ms=16, mfc='blue', mec='blue')

    pl.text(.35, 10**-.4, r'$10^7 M_\odot$', color='k', fontsize=14)
    pl.text(.35, 10 ** -.5, r'$10^6 M_\odot$', color='blueviolet', fontsize=14)
    pl.text(.35, 10 ** -.6, r'$10^5 M_\odot$', color='blue', fontsize=14)

    ylab1 = [1.]
    ylabti1 = ['1.0']
    ylab2 = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    ylabti2 = ['', '', '', '', '0.5', '', '', '', '']
    ax.set_yticks(ylab1)
    ax.set_yticklabels(ylabti1)
    ax.set_yticks(ylab2, minor=True)
    ax.set_yticklabels(ylabti2, minor=True)
    plt.tick_params(which='minor', length=4, color='k')

    xlab1 = [1., 10.]
    xlabti1 = ['1.0', '10.0']
    xlab2 = [.3,.4,.5,.6,.7,.8,.9, 2.,3.,4.,5.,6.,7.,8.,9.,]
    xlabti2 = ['', '', '0.5', '', '', '', '','', '', '', '','', '', '', '']
    ax.set_xticks(xlab1)
    ax.set_xticklabels(xlabti1)
    ax.set_xticks(xlab2, minor=True)
    ax.set_xticklabels(xlabti2, minor=True)

    folder = MAIN_PATH + "/SubhaloDetection/Plots/"
    fig_name = folder + 'Sigma68.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def dmax_extend(cs=2.2*10**-26., annih='BB'):
    hwlist = []
    modellist = []
    hwlist2 = []
    modellist2 = []
    hwlist3 = []
    modellist3 = []
    m_full = np.logspace(5., 7., 100)
    masslist = np.logspace(5., 7., 6)
    for i, m in enumerate(masslist):
        hwlist.append(HW_Fit(m))
        modellist.append(Model(100, cs, annih, m, profile=2,
                               m200=False, rb=hwlist[i].rb))
        hwlist2.append(HW_Fit(m))
        modellist2.append(Model(10, cs, annih, m, profile=2,
                               m200=False, rb=hwlist2[i].rb))
        hwlist3.append(HW_Fit(m))
        modellist3.append(Model(1000, cs, annih, m, profile=2,
                               m200=False, rb=hwlist3[i].rb))

    dmax_ex_tab = np.zeros(masslist.size * 2).reshape((masslist.size, 2))
    dmax_p_tab = np.zeros(masslist.size * 2).reshape((masslist.size, 2))
    dmax_ex_tab2 = np.zeros(masslist.size * 2).reshape((masslist.size, 2))
    dmax_p_tab2 = np.zeros(masslist.size * 2).reshape((masslist.size, 2))
    dmax_ex_tab3 = np.zeros(masslist.size * 2).reshape((masslist.size, 2))
    dmax_p_tab3 = np.zeros(masslist.size * 2).reshape((masslist.size, 2))
    # for i in range(masslist.size):
    #     print i + 1, '/', len(masslist)
    #     dmax_ex_tab[i] = [masslist[i], modellist[i].D_max_extend()]
    #     dmax_p_tab[i] = [masslist[i], modellist[i].d_max_point()]
    #     dmax_ex_tab2[i] = [masslist[i], modellist2[i].D_max_extend()]
    #     dmax_p_tab2[i] = [masslist[i], modellist2[i].d_max_point()]
    #     dmax_ex_tab3[i] = [masslist[i], modellist3[i].D_max_extend()]
    #     dmax_p_tab3[i] = [masslist[i], modellist3[i].d_max_point()]

    dmax_ex_tab = np.array([[1.00000000e+05, 5.78432934e-01], [2.51188643e+05, 8.12183406e-01],
                            [6.30957344e+05, 1.14039188e+00], [1.58489319e+06, 1.60122327e+00],
                            [3.98107171e+06, 2.24829549e+00], [1.00000000e+07, 3.15683745e+00]])
    dmax_p_tab = np.array([[1.00000000e+05, 6.91368288e-01], [2.51188643e+05, 9.70754596e-01],
                            [6.30957344e+05, 1.36304268e+00], [1.58489319e+06, 1.91385687e+00],
                            [3.98107171e+06, 2.68725856e+00], [1.00000000e+07, 3.77319678e+00]])
    dmax_ex_tab2 = np.array([[1.00000000e+05, 1.28564966e+00], [2.51188643e+05, 1.80518302e+00],
                            [6.30957344e+05, 2.53466337e+00], [1.58489319e+06, 3.55895722e+00],
                            [3.98107171e+06, 4.99714466e+00], [1.00000000e+07, 7.01650060e+00]])
    dmax_p_tab2 = np.array([[1.00000000e+05, 1.53666008e+00], [2.51188643e+05, 2.15763416e+00],
                            [6.30957344e+05, 3.02954780e+00], [1.58489319e+06,4.25380727e+00],
                            [3.98107171e+06,5.97279776e+00], [1.00000000e+07, 8.38644319e+00]])
    dmax_ex_tab3 = np.array([[1.00000000e+05, 1.05796427e-01], [2.51188643e+05, 1.48550072e-01],
                            [6.30957344e+05, 2.08579389e-01], [1.58489319e+06, 2.92867013e-01],
                            [3.98107171e+06,  4.11217399e-01], [1.00000000e+07, 5.77393099e-01]])
    dmax_p_tab3 = np.array([[1.00000000e+05, 1.26452585e-01], [2.51188643e+05, 1.77552877e-01],
                            [6.30957344e+05, 2.49303120e-01], [1.58489319e+06, 3.50048091e-01],
                            [3.98107171e+06,  4.91504743e-01], [1.00000000e+07, 6.90124925e-01]])

    ex_plot = np.polyfit(np.log10(dmax_ex_tab[:, 0]), np.log10(dmax_ex_tab[:, 1]), 1)
    p_plot = np.polyfit(np.log10(dmax_p_tab[:, 0]), np.log10(dmax_p_tab[:, 1]), 1)
    ex_plot2 = np.polyfit(np.log10(dmax_ex_tab2[:, 0]), np.log10(dmax_ex_tab2[:, 1]), 1)
    p_plot2 = np.polyfit(np.log10(dmax_p_tab2[:, 0]), np.log10(dmax_p_tab2[:, 1]), 1)
    ex_plot3 = np.polyfit(np.log10(dmax_ex_tab3[:, 0]), np.log10(dmax_ex_tab3[:, 1]), 1)
    p_plot3 = np.polyfit(np.log10(dmax_p_tab3[:, 0]), np.log10(dmax_p_tab3[:, 1]), 1)

    ex_int = 10. ** np.polyval(ex_plot, np.log10(m_full))
    p_int = 10. ** np.polyval(p_plot, np.log10(m_full))
    ex_int2 = 10. ** np.polyval(ex_plot2, np.log10(m_full))
    p_int2 = 10. ** np.polyval(p_plot2, np.log10(m_full))
    ex_int3 = 10. ** np.polyval(ex_plot3, np.log10(m_full))
    p_int3 = 10. ** np.polyval(p_plot3, np.log10(m_full))

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    ymin = np.min([ex_int[0], ex_int2[0], ex_int3[0]])
    ymax = np.max([p_int[-1], p_int2[-1], p_int3[-1]])
    pl.xlim([10 ** 5., 10. ** 7.])
    pl.ylim([ymin, ymax])
    pl.xlabel(r'Mass  [$M_\odot$]', fontsize=20)
    pl.ylabel(r'$D_{\rm max}$   [kpc]', fontsize=20)
    pl.plot(m_full, ex_int, '--', lw=2, color='red', dashes=(10,20))
    pl.plot(m_full, p_int, lw=2, color='red')
    pl.plot(m_full, ex_int2, '--', lw=2, color='goldenrod', dashes=(10, 20))
    pl.plot(m_full, p_int2, lw=2, color='goldenrod')
    pl.plot(m_full, ex_int3, '--', lw=2, color='maroon', dashes=(10, 20))
    pl.plot(m_full, p_int3, lw=2, color='maroon')
    pl.text(1.5 * 10**5., ymax * .8, r'$\chi \chi$' + r'$\rightarrow$' + r'$b\bar{b}$', color='k', fontsize=14)
    ylab1 = [1.]
    ylabti1 = ['1.0']
    ylab2 = [.5, 5.]
    ylabti2 = ['0.5','5.0']
    ax.set_yticks(ylab1)
    ax.set_yticklabels(ylabti1)
    ax.set_yticks(ylab2, minor=True)
    ax.set_yticklabels(ylabti2, minor=True)
    plt.tick_params(which='minor', length=4, color='k')
    folder = MAIN_PATH + "/SubhaloDetection/Plots/"
    fig_name = folder + 'Dmax_Ext_vs_Plike.pdf'
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def limit_comparison(plike=True, bmin=20., annih_prod='BB', nobs=True):
    if plike:
        ptag = 'Pointlike'
    else:
        ptag = 'Extended'
    if nobs:
        ntag = ''
    else:
        ntag = '_Nobs_False_'
    dir = MAIN_PATH + '/SubhaloDetection/Data/'
    # nfw = np.loadtxt(dir + 'Limits_' + ptag + '_NFW_Truncate_False_alpha_0.16_annih_prod_' +
    #                  annih_prod + '_arxiv_num_160106781_bmin_{:.1f}'.format(bmin) +
    #                  '_Mlow_4.000_' + ntag + '.dat')
    # blh_old = np.loadtxt(dir + 'Limits_' + ptag + '_Einasto_Truncate_True_alpha_0.16_annih_prod_' +
    #                  annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
    #                  '_Mlow_0.000_OLD' + ntag + '.dat')
    # blh_new = np.loadtxt(dir + 'Limits_' + ptag + '_Einasto_Truncate_True_alpha_0.16_annih_prod_' +
    #                      annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
    #                      '_Mlow_0.000_NEW' + ntag + '.dat')
    hw_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_True_.dat')
    hw_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')
    hw_low_null = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                        annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_False_.dat')
    hw_high_null = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_False_.dat')

    list = [hw_low, hw_high, hw_low_null, hw_high_null]
    color_list = ['blueviolet', 'blueviolet', 'black', 'black']

    mass_list = np.logspace(1., 3., 100)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** 1., 10. ** 3.])
    pl.ylim([10 ** -27., 10. ** -23.])
    for i, lim in enumerate(list):
        pts = interpola(mass_list, lim[:, 0], lim[:, 1])
        plt.plot(mass_list, pts, lw=1, color=color_list[i], alpha=0.3)

    lowm2 = interpola(mass_list, hw_low_null[:, 0], hw_low_null[:, 1])
    highm2 = interpola(mass_list, hw_high_null[:, 0], hw_high_null[:, 1])
    ax.fill_between(mass_list, lowm2, highm2, where=highm2 >= lowm2,
                    facecolor='black', edgecolor='None', interpolate=True, alpha=0.3)
    lowm = interpola(mass_list, hw_low[:, 0], hw_low[:, 1])
    highm = interpola(mass_list, hw_high[:, 0], hw_high[:, 1])
    ax.fill_between(mass_list, lowm, highm, where=highm >= lowm,
                    facecolor='blueviolet', edgecolor='None', interpolate=True, alpha=0.3)
    ltop = 10 ** -23.4
    ldwn = 10 ** -.3
    #plt.text(15, ltop, 'NFW (Post-tidal Stripping)', color='Magenta', fontsize=10)
    #plt.text(15, ltop * ldwn, 'Einasto (Pre-tidal Stripping)', color='k', fontsize=10)
    #plt.text(15, ltop * ldwn ** 2., 'This Work', color='blue', fontsize=10)
    plt.text(15, ltop, r'$\chi \chi$ ' + r'$\rightarrow$' + r' $b \bar{{b}}$', color='k', fontsize=16)
    plt.text(15, ltop * ldwn, r'Point-like', color='k', fontsize=12)
    plt.axhline(y=2.2 * 10 **-26., xmin=0, xmax=1, lw=1, ls='--', color='k', alpha=1)
    figname = dir + 'Limit_Comparison_' + ptag +'annih_prod_' + \
              annih_prod + '_bmin_{:.0f}'.format(bmin) + ntag + '.pdf'

    pl.xlabel(r'$m_\chi$   [GeV]', fontsize=20)
    pl.ylabel(r'$\left< \sigma v \right>$   [$\textrm{cm}^3 \textrm{s}^{{-1}}$]', fontsize=20)
    fig.set_tight_layout(True)
    pl.savefig(figname)
    return


def limit_comparison_extended(bmin=20., annih_prod='BB', nobs=False):
    ptag = 'Extended'
    if nobs:
        ntag = ''
    else:
        ntag = '_Nobs_False_'
    dir = MAIN_PATH + '/SubhaloDetection/Data/'

    hw_high = np.loadtxt(dir + 'Limits_Extended_HW_Truncate_False_alpha_0.16_annih_prod_'
                               'BB_arxiv_num_13131729_bmin_20.0_Mlow_5.000__Nobs_False_.dat')
    hw_p = np.loadtxt(dir + 'Limits_Pointlike_HW_Truncate_False_alpha_0.16_annih_prod_' +
                              annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                              '_Mlow_5.000__Nobs_True_.dat')

    color_list = ['green', 'black']

    mass_list = np.logspace(1., 3., 100)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** 1., 10. ** 3.])
    pl.ylim([10 ** -27., 10. ** -23.])

    pts = interpola(mass_list, hw_high[:, 0], hw_high[:, 1])
    plt.plot(mass_list, pts, lw=2, color=color_list[0], alpha=1)
    hp = interpola(mass_list, hw_p[:, 0], hw_p[:, 1])
    plt.plot(mass_list, hp, '--', lw=2, color='k', alpha=1, dashes=(20, 15))

    ltop = 10 ** -23.4
    ldwn = 10 ** -.3
    plt.text(15, ltop, r'$\chi \chi$ ' + r'$\rightarrow$' + r' $b \bar{{b}}$', color='k', fontsize=16)
    plt.text(15, ltop * ldwn ** 2., r'$\sigma_{{68}} > 0.31^\circ$', color='green', fontsize=16)
    plt.axhline(y=2.2 * 10 **-26., xmin=0, xmax=1, lw=1, ls='--', color='k', alpha=1)
    figname = dir + 'Limit_Comparison_' + ptag +'annih_prod_' + \
              annih_prod + '_bmin_{:.0f}'.format(bmin) + ntag + '.pdf'
    plt.text(15, ltop * ldwn, r'$M_{{min}} = 10^{{5}} M_\odot$', color='k', fontsize=16)
    pl.xlabel(r'$m_\chi$   [GeV]', fontsize=20)
    pl.ylabel(r'$\left< \sigma v \right>$   [$\textrm{cm}^3 \textrm{s}^{{-1}}$]', fontsize=20)
    fig.set_tight_layout(True)
    pl.savefig(figname)
    return


def limit_comparison_all(plike=True, bmin=20., nobs=True):
    if plike:
        ptag = 'Pointlike'
    else:
        ptag = 'Extended'
    if nobs:
        ntag = ''
    else:
        ntag = '_Nobs_False_'
    dir = MAIN_PATH + '/SubhaloDetection/Data/'
    bb_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         'BB' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_True_.dat')
    bb_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         'BB' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')
    cc_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                        'CC' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_True_.dat')
    cc_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         'CC' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')
    zz_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                        'ZZ' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_True_.dat')
    zz_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         'ZZ' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')
    ww_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                        'WW' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_True_.dat')
    ww_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         'WW' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')
    tautau_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                        'tautau' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000__Nobs_True_.dat')
    tautau_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         'tautau' + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')


    list = [bb_low, bb_high, cc_low, cc_high, zz_low, zz_high,
            ww_low, ww_high, tautau_low, tautau_high]
    color_list = ['blueviolet', 'goldenrod',
                  'maroon', 'blue', 'green']
    ls_list = ['-', '--', ':', '-.', '--']

    mass_list = np.logspace(1., 3., 100)
    mz_list = np.logspace(np.log10(91.), 3., 100)
    mw_list = np.logspace(np.log10(80.), 3., 100)

    f, ax = plt.subplots(1, 2, sharey='col', figsize=(10, 5))

    ax[0].set_xscale("log")
    ax[0].set_yscale('log')
    ax[0].set_xlim([10 ** 1., 10. ** 3.])
    ax[0].set_ylim([2 *10 ** -27., 10. ** -24.])
    ax[1].set_xscale("log")
    ax[1].set_yscale('log')
    ax[1].set_xlim([10 ** 1., 10. ** 3.])
    ax[1].set_ylim([2 * 10 ** -27., 10. ** -24.])
    for i in range(len(list) / 2):
        if i == 2:
            m = mz_list
        elif i == 3:
            m = mw_list
        else:
            m = mass_list
        pts1 = interpola(m, list[2 * i][:, 0], list[2 * i][:, 1])
        pts2 = interpola(m, list[2 * i + 1][:, 0], list[2 * i + 1][:, 1])
        #ax.fill_between(m, pts1, pts2, where=pts2 >= pts1,
        #                facecolor=color_list[i], edgecolor='None', interpolate=True, alpha=0.3)

        ax[0].plot(m, pts1, color=color_list[i], lw=2)
        ax[1].plot(m, pts2, color=color_list[i], lw=2)

    ltop = 10 ** -24.4
    ldwn = 10 ** -.2
    ax[1].text(15, ltop, r'$b\bar{{b}}$', color='blueviolet', fontsize=18)
    ax[1].text(15, ltop * ldwn, r'$c\bar{{c}}$', color='goldenrod', fontsize=18)
    ax[1].text(15, ltop * ldwn ** 2.,  r'$ZZ$', color='maroon', fontsize=18)
    ax[1].text(15, ltop * ldwn ** 3., r'$W^+W^-$', color='blue', fontsize=18)
    ax[1].text(15, ltop * ldwn ** 4., r'$\tau^+\tau^-$', color='green', fontsize=18)
    ax[0].text(15, ltop, r'$b\bar{{b}}$', color='blueviolet', fontsize=18)
    ax[0].text(15, ltop * ldwn, r'$c\bar{{c}}$', color='goldenrod', fontsize=18)
    ax[0].text(15, ltop * ldwn ** 2., r'$ZZ$', color='maroon', fontsize=18)
    ax[0].text(15, ltop * ldwn ** 3., r'$W^+W^-$', color='blue', fontsize=18)
    ax[0].text(15, ltop * ldwn ** 4., r'$\tau^+\tau^-$', color='green', fontsize=18)

    ax[0].text(150, 3 * 10**-27., r'$M_{{min}} = 10^{{-5}} M_\odot$', color='k', fontsize=16)
    ax[1].text(150, 3 * 10 ** -27., r'$M_{{min}} = 10^{{5}} M_\odot$', color='k', fontsize=16)
    #plt.text(15, ltop, r'$\chi \chi$ ' + r'$\rightarrow$' + r' $b \bar{{b}}$', color='k', fontsize=16)
    #plt.text(15, ltop * ldwn, r'Point-like', color='k', fontsize=12)
    ax[1].axhline(y=2.2 * 10 **-26., xmin=0, xmax=1, lw=1, ls='--', color='k', alpha=1)
    ax[0].axhline(y=2.2 * 10 ** -26., xmin=0, xmax=1, lw=1, ls='--', color='k', alpha=1)



    figname = dir + 'Limit_Comparison_' + ptag +\
              'Annih_to_All' + '_bmin_{:.0f}'.format(bmin) + ntag + '.pdf'

    ax[0].set_xlabel(r'$m_\chi$   [GeV]', fontsize=20)
    ax[1].set_xlabel(r'$m_\chi$   [GeV]', fontsize=20)
    ax[1].yaxis.tick_right()
    ax[0].set_ylabel(r'$\left< \sigma v \right>$   [$\textrm{cm}^3 \textrm{s}^{{-1}}$]', fontsize=20)
    #ax[1].set_ylabel(r'$\left< \sigma v \right>$   [$cm^3 s^{{-1}}$]', fontsize=20)
    f.tight_layout(rect=(0, .0, 1, 1))
    plt.subplots_adjust(wspace=0.1, hspace=0.)
    pl.savefig(figname)
    return



def obtain_number_density():
    max_mass = 1 * 10. ** 10.
    min_mass = 1 * 10. ** 8.
    gcd_range = np.array([0., 300.])
    n_halos = 24. + 24. + 3.
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ELVIS_Halo_Catalogs/'
    file_hi = np.loadtxt(dir + 'Hi_Res_Subhalos.dat')
    file_iso = np.loadtxt(dir + 'Iso_Subhalos.dat')
    file_pair = np.loadtxt(dir + 'Paired_Subhalos.dat')
    ind_of_int = []
    gcd1 = file_hi[0, 1:4]
    gcd2 = file_iso[0, 1:4]
    for i, sub in enumerate(file_hi):
        if i != 0:
            dist1 = np.sqrt(((gcd1 - sub[1:4]) ** 2.).sum()) * 1000.
            sub[1] = dist1
            if dist1 < 300.:
                ind_of_int.append(i)
    file_hi = file_hi[ind_of_int]
    ind_of_int = []
    for i, sub in enumerate(file_iso):
        if i != 0:
            dist1 = np.sqrt(((gcd2 - sub[1:4]) ** 2.).sum()) * 1000.
            sub[1] = dist1
            if dist1 < 300.:
                ind_of_int.append(i)
    file_iso = file_iso[ind_of_int]

    gcd3 = file_pair[0, 1:4]
    gcd3_2 = file_pair[1, 1:4]
    ind_of_int = []
    for i, sub in enumerate(file_pair):
        if i != 0 and i != 1:
            dist1 = np.sqrt(((gcd3 - sub[1:4]) ** 2.).sum()) * 1000.
            dist2 = np.sqrt(((gcd3_2 - sub[1:4]) ** 2.).sum()) * 1000.
            sub[1] = np.min([dist1, dist2])
            if sub[1] < 300.:
                ind_of_int.append(i)
    file_pair = file_pair[ind_of_int]
    subhalos = np.vstack((file_iso[:, :10], file_hi[:, :10], file_pair[:, :10]))
    subhalos = subhalos[subhalos[:, 1] != 0.]
    print 'Num Subhalos: ', len(subhalos)
    fig = plt.figure(figsize=(8., 6.))
    mass_bins = np.logspace(np.log10(4. * 10.**6.), 10.,
                            (10. - np.log10(3. * 10.**6.)) * 15)
    print 'Making Subhalo Mass Number Density Histogram...'

    plt.hist(subhalos[:, 9], bins=mass_bins, log=True, normed=False,
             color='k', weights=1. / subhalos[:, 9], histtype='step', lw=2)

    fit_einw, binew = np.histogram(subhalos[:, 9], bins=mass_bins, normed=False, weights=1. / subhalos[:, 9])
    d_einw = np.zeros(len(mass_bins) - 1)
    for i in range(len(mass_bins) - 1):
        d_einw[i] = np.median(subhalos[(subhalos[:, 9] < mass_bins[i + 1]) &
                                      (subhalos[:, 9] > mass_bins[i])][:, 9])

    parmsw, covw = curve_fit(lin_fit, np.log10(d_einw), np.log10(fit_einw))
    plot_einw = lin_fit(np.log10(mass_bins), parmsw[0], parmsw[1])
    plt.plot(mass_bins, 10. ** plot_einw, color='blue', lw=2)
    plt.text(1*10 ** 7., 10. ** -3.8, r'$\frac{{d N}}{{d M}} \propto M^{{{:.1f}}}$'.format(parmsw[0]), fontsize=14,
             color='b')

    pl.gca().set_xscale("log")
    pl.xlim([4. * 10.**6., 10.**10.])
    plt.xlabel(r'Mass   [$M_\odot$]', fontsize=18)
    plt.ylabel(r'$\frac{dN}{dM}$', fontsize=22, rotation='horizontal')
    subhalos = subhalos[(subhalos[:, 9] < max_mass) & (subhalos[:, 9] > min_mass)]
    mass_bins = np.logspace(np.log10(np.min(subhalos[:, 9])), np.log10(np.max(subhalos[:, 9])),
                            (np.log10(np.max(subhalos[:, 9])) - np.log10(np.min(subhalos[:, 9]))) * 15)
    fit_ein, bine = np.histogram(subhalos[:, 9], bins=mass_bins, normed=False, weights=1. / subhalos[:, 9])
    d_ein = np.zeros(len(mass_bins) - 1)

    for i in range(len(mass_bins) - 1):
            d_ein[i] = np.median(subhalos[(subhalos[:, 9] < mass_bins[i + 1]) &
                                          (subhalos[:, 9] > mass_bins[i])][:, 9])

    parms, cov = curve_fit(lin_fit, np.log10(d_ein), np.log10(fit_ein))
    plot_ein = lin_fit(np.log10(mass_bins), parms[0], parms[1])
    plt.plot(mass_bins, 10. ** plot_ein, color='red', lw=2)
    plt.text(2 * 10 ** 8., 10. ** -4.5, 'Fit Range: [$10^{8}, 10^{10}$]' + ' $M_\odot$',
             fontsize=14, color='r')
    plt.text(7. * 10 ** 6., 10. ** -7., 'ELVIS Subhalos: \n  $r \leq 300$ kpc',
             fontsize=14, color='k')

    plt.text(2 * 10 ** 8., 10. ** -5., r'$\frac{{d N}}{{d M}} \propto M^{{{:.1f}}}$'.format(parms[0]), fontsize=14,
             color='red')

    fname = 'Subhalo_Mass_Density.pdf'
    pl.savefig(dir + '../../../Plots/' + fname)

    dist_bins = np.linspace(1., 300., 20)
    print 'Making dN / dM Histogram...'
    fig = plt.figure(figsize=(8., 6.))

    plt.hist(subhalos[:, 1], bins=dist_bins, log=True, normed=False,
             weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.),
             color='k', histtype='step', lw=2)

    pl.gca().set_xscale("log")
    pl.xlim([10., 300])

    plt.xlabel(r'Distance   [kpc]', fontsize=18)
    plt.ylabel(r'$\frac{dN}{dV}$', fontsize=22, rotation='horizontal')

    fit_ein, bine = np.histogram(subhalos[:, 1], bins=dist_bins, normed=False,
                                 weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.))
    d_ein = np.zeros(len(dist_bins) - 1)
    for i in range(len(dist_bins) - 1):
        d_ein[i] = np.median(subhalos[(subhalos[:, 1] < dist_bins[i + 1]) &
                                      (subhalos[:, 1] > dist_bins[i])][:, 1])

    parms, cov = curve_fit(einasto_fit, d_ein, fit_ein, bounds=([30., 0.2, 0.], [300., .7, .1]), sigma=fit_ein)
    lden = einasto_fit(8.5, parms[0], parms[1], parms[2])
    #print parms
    print 'Local: ', lden
    hist_norm = 0.9 * lden / (n_halos * (min_mass ** (-0.9) - max_mass ** (-0.9)))
    #hist_norm = .8 * lden / (n_halos * (min_mass ** (-.8) - max_mass ** (-.8)))
    plot_ein = einasto_fit(dist_bins, parms[0], parms[1], parms[2])
    plt.plot(dist_bins, plot_ein, color='red', lw=2)
    plt.text(12., 10. ** -2.3, 'Fit Range: [$10^{8}, 10^{10}$]' + ' $M_\odot$',
             fontsize=14, color='k')

    plt.text(100., 10. ** -2.4, 'Einasto Parameters:', fontsize=14, color='r')
    plt.text(100., 10. ** -2.7, r'$\alpha = {:.2f}$'.format(parms[1]), fontsize=14, color='r')
    plt.text(100., 10. ** -3., r'$r_s = {:.2f}$'.format(parms[0]), fontsize=14, color='r')
    plt.text(12., 10. ** -4.5, r'$\frac{{d N}}{{dM dV}} = \frac{{{:.0f}}}{{kpc^{{3}}}} $'.format(hist_norm) +
             r'$\left(\frac{M}{M_\odot}\right)^{-1.9}$', fontsize=14, color='k')

    fname = 'NumberDensity_Histogram_ELVIS_AlternateExponent.pdf'
    pl.savefig(dir + '../../../Plots/' + fname)

    return


def scatter_fits():
    m_list = [4.10e+05, 5.35e+06, 1.70e+07, 3.16e+07, 5.52e+07, 1.18e+08, 10.**10.]
    gcd_range = np.linspace(0., 300., 100)
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    sim_list = ['ViaLacteaII_Useable_Subhalos.dat', 'Elvis_Useable_Subhalos.dat']
    class_list = [Via_Lactea_II(), ELVIS()]
    subs = [np.loadtxt(dir + sim_list[0]),
                     np.loadtxt(dir + sim_list[1])]
    names = ['Via Lactea II', 'ELVIS']

    f, ax = plt.subplots(6, 2, sharex='col')
    for j in range(len(m_list) - 1):
        mlow, mhigh = m_list[j], m_list[j + 1]
        print 'Mrange: [{:.2e} {:.2e}]'.format(mlow, mhigh)
        for i in range(len(class_list)):
            print names[i]
            sub_o_int = class_list[i].find_subhalos(min_mass=mlow, max_mass=mhigh,
                                                         gcd_min=gcd_range[0],
                                                         gcd_max=gcd_range[-1],
                                                         print_info=False)
            if i == 0:
                subhalos = subs[i][sub_o_int][:, :6]
            else:
                subhalos = np.vstack((subhalos, subs[i][sub_o_int][:, :6]))

        print 'Calculating Fits...'
        for k, sub in enumerate(subhalos):
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
        r_groups = Joint_Simulation_Comparison().equipartition(x=gamma_plt[:, 0], bnd_num=regions)
        r_bounds = [r_groups[a][0] for a in range(regions)]
        r_bounds.append(r_groups[-1][-1])
        print 'GCD Bounds: ', r_bounds
        r_diff_tab = np.diff(r_bounds) / 2. + r_bounds[:-1]
        fit_tabs_g = np.zeros(regions * 7).reshape((regions, 7))
        fit_tabs_rb = np.zeros(regions * 5).reshape((regions, 5))

        for i in range(regions):

            print 'Dist Range: ', r_bounds[i], r_bounds[i + 1]
            args_o_int = (r_bounds[i] < gamma_plt[:, 0]) & (gamma_plt[:, 0] < r_bounds[i + 1])
            print 'Number Subhalos in Range: ', args_o_int.sum()
            print 'Mean Gamma of Subhalos in Range: ', np.mean(gamma_plt[args_o_int][:, 1])
            r_diff = np.median(gamma_plt[args_o_int][:, 0])

            hist_g, bin_edges_g = np.histogram(gamma_plt[args_o_int][:, 1], bins='auto', normed=True)
            x_fit_g = np.diff(bin_edges_g) / 2. + bin_edges_g[:-1]
            mu_g, sig_g, k_g = gen_norm_fit_finder(x_fit_g, hist_g, sigma=np.sqrt(hist_g), mu_low=.5,
                                                   mu_high=2., klow=.01, khigh=1., slow=.1, shigh=1.)
            median_g = mu_g
            std_dev_g_low = lower_sigma_gen_norm(mu_g, sig_g, k_g)
            std_dev_g_high = upper_sigma_gen_norm(mu_g, sig_g, k_g)
            fit_tabs_g[i] = [r_diff, mu_g, sig_g, k_g, median_g,
                             std_dev_g_low, std_dev_g_high]

            hist_rb, bin_edges_rb = np.histogram(rb_plt[args_o_int][:, 1], bins='auto', normed=True)
            x_fit_rb = np.diff(bin_edges_rb) / 2. + bin_edges_rb[:-1]
            pars1, cov = curve_fit(lnnormal, x_fit_rb, hist_rb, bounds=(0., np.inf))
            mu, sig = [pars1[0], pars1[1]]
            std_dev_r_high = lower_sigma_lnnorm(mu, sig)
            std_dev_r_low = upper_sigma_lnnorm(mu, sig)
            fit_tabs_rb[i] = [r_diff, mu, sig, std_dev_r_low, std_dev_r_high]

        gam_mean_fit = np.polyfit(np.log10(fit_tabs_g[:, 0]), np.log10(fit_tabs_g[:, 4]), 1)
        gcd_tab_plt = np.logspace(1., np.log10(300.), 100)
        gam_fit_plt = 10. ** np.polyval(gam_mean_fit, np.log10(gcd_tab_plt))

        ax[j, 0].set_xscale("log")
        ax[j, 0].set_yscale('log')
        ax[j, 0].set_xlim([10., 300.])
        ymin = 0.3
        ymax = np.max(gamma_plt[:, 1])
        ax[j, 0].set_ylim([ymin, ymax])
        if j == len(m_list) - 1:
            ax[j, 0].set_xlabel('GC Distance (kpc)', fontsize=16)
        ax[j, 0].set_ylabel(r'$\gamma$', fontsize=16)

        ax[j, 0].plot(gamma_plt[:, 0], gamma_plt[:, 1], 'ro', alpha=0.3)
        ax[j, 0].plot(fit_tabs_g[:, 0], fit_tabs_g[:, 4], 'kx', ms=6, mew=3)

        for x in range(fit_tabs_g[:, 0].size):
            if (fit_tabs_g[x, 4] - fit_tabs_g[x, 5]) > 0.:
                yerr_l = (np.log10(fit_tabs_g[x, 4] - fit_tabs_g[x, 5]) -
                          np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            else:
                yerr_l = 0.
            yerr_h = (np.log10(fit_tabs_g[x, 4] + fit_tabs_g[x, 6]) -
                      np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))

            ax[j, 0].axvline(x=fit_tabs_g[x, 0], ymin=yerr_l, ymax=yerr_h,
                            linewidth=2, color='Black')

        ymin = np.min(gamma_plt[:, 1])
        ymax = np.max(gamma_plt[:, 1])
        legtop = .8 * (ymax - ymin)
        ax[j, 0].plot(gcd_tab_plt, gam_fit_plt, 'k', ms=2)
        ax[j, 0].text(12., legtop, r'$\gamma(R_\oplus)$ = {:.3f}'.format(10. ** np.polyval(gam_mean_fit, np.log10(8.5))),
                      fontsize=16, ha='left', va='center', color='Black')

        print 'Making Rb vs GCD scatter...'
        rb_mean_fit = np.polyfit(np.log10(fit_tabs_rb[:, 0]), np.log10(fit_tabs_rb[:, 1]), 1)
        rb_fit_plt = 10. ** np.polyval(rb_mean_fit, np.log10(gcd_tab_plt))

        ax[j, 1].set_xscale("log")
        ax[j, 1].set_yscale('log')
        ax[j, 1].set_xlim([10., 300.])
        ymin = np.min(rb_plt[:, 1])
        ymax = np.max(rb_plt[:, 1])
        ax[j, 1].set_ylim([ymin, ymax])
        if j == len(m_list) - 1:
            ax[j, 1].set_xlabel('GC Distance (kpc)', fontsize=20)
        ax[j, 1].set_ylabel(r'$R_b$   [kpc]', fontsize=20)
        ax[j, 1].plot(rb_plt[:, 0], rb_plt[:, 1], 'ro', alpha=0.3)
        ax[j, 1].plot(fit_tabs_rb[:, 0], fit_tabs_rb[:, 1], 'kx', ms=6, mew=3)
        ax[j, 1].plot(gcd_tab_plt, rb_fit_plt, 'k', ms=2)

        for x in range(fit_tabs_rb[:, 0].size):
            if (fit_tabs_rb[x, 1] - fit_tabs_rb[x, 3]) > 0.:
                yerr_l = (np.log10(fit_tabs_rb[x, 1] - fit_tabs_rb[x, 3]) -
                          np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            else:
                yerr_l = 0.
            yerr_h = (np.log10(fit_tabs_rb[x, 1] + fit_tabs_rb[x, 4]) -
                      np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            ax[j, 1].axvline(x=fit_tabs_rb[x, 0], ymin=yerr_l, ymax=yerr_h,
                        linewidth=2, color='Black')
        legtop = .8 * (ymax - ymin)
        legdown = .1 * (ymax - ymin)
        ax[j, 1].text(12., legtop, r'$R_b(R_\oplus)$ = {:.3f}'.format(10. ** np.polyval(rb_mean_fit, np.log10(8.5))),
                 fontsize=16, ha='left', va='center', color='Black')
        #ax[j, 1].text(12., legtop - legdown, r'$R_b(R_\oplus)$ = {:.3f}'.format(10. ** np.polyval(rb_mean_fit, np.log10(8.5))),
        #              fontsize=16, ha='left', va='center', color='Black')

    f.tight_layout(rect=(0, .0, 1, 1))
    plt.subplots_adjust(wspace=0.15, hspace=0.)
    figname = MAIN_PATH + '/SubhaloDetection/Plots/PaperPlots/Scatter_Multiplot.pdf'
    pl.savefig(figname)



def uncertainty():
    max_mass = 1 * 10. ** 10.
    min_mass = 1 * 10. ** 8.
    gcd_range = np.array([0., 300.])
    n_halos = 24. + 24. + 3.
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ELVIS_Halo_Catalogs/'
    file_hi = np.loadtxt(dir + 'Hi_Res_Subhalos.dat')
    file_iso = np.loadtxt(dir + 'Iso_Subhalos.dat')
    file_pair = np.loadtxt(dir + 'Paired_Subhalos.dat')
    ind_of_int = []
    gcd1 = file_hi[0, 1:4]
    gcd2 = file_iso[0, 1:4]
    for i, sub in enumerate(file_hi):
        if i != 0:
            dist1 = np.sqrt(((gcd1 - sub[1:4]) ** 2.).sum()) * 1000.
            sub[1] = dist1
            if dist1 < 300.:
                ind_of_int.append(i)
    file_hi = file_hi[ind_of_int]
    ind_of_int = []
    for i, sub in enumerate(file_iso):
        if i != 0:
            dist1 = np.sqrt(((gcd2 - sub[1:4]) ** 2.).sum()) * 1000.
            sub[1] = dist1
            if dist1 < 300.:
                ind_of_int.append(i)
    file_iso = file_iso[ind_of_int]

    gcd3 = file_pair[0, 1:4]
    gcd3_2 = file_pair[1, 1:4]
    ind_of_int = []
    for i, sub in enumerate(file_pair):
        if i != 0 and i != 1:
            dist1 = np.sqrt(((gcd3 - sub[1:4]) ** 2.).sum()) * 1000.
            dist2 = np.sqrt(((gcd3_2 - sub[1:4]) ** 2.).sum()) * 1000.
            sub[1] = np.min([dist1, dist2])
            if sub[1] < 300.:
                ind_of_int.append(i)
    file_pair = file_pair[ind_of_int]
    subhalos = np.vstack((file_iso[:, :10], file_hi[:, :10], file_pair[:, :10]))
    subhalos = subhalos[subhalos[:, 1] != 0.]
    print 'Num Subhalos: ', len(subhalos)
    fig = plt.figure(figsize=(8., 6.))
    mass_bins = np.logspace(np.log10(4. * 10.**6.), 10.,
                            (10. - np.log10(3. * 10.**6.)) * 15)
    print 'Making Subhalo Mass Number Density Histogram...'

    plt.hist(subhalos[:, 9], bins=mass_bins, log=True, normed=False,
             color='k', weights=1. / subhalos[:, 9], histtype='step', lw=2)

    pl.gca().set_xscale("log")
    pl.xlim([1. * 10.**8., 10.**10.])
    pl.ylim([10**-10, 10**-3])
    plt.xlabel(r'Mass   [$M_\odot$]', fontsize=18)
    plt.ylabel(r'$\frac{dN}{dM}$', fontsize=22, rotation='horizontal')

    beta_tab = np.array([89., 237., 628., 1660., 4378.])
    alpha_tab = np.array([-1.8, -1.85, -1.9, -1.95, -2.])
    color_tab = ['red','orange','yellow','green','blue']

    for i in range(beta_tab.size):
        mass_list = np.logspace(8, 10, 100)
        #plt_info = beta_tab[i] * mass_list ** alpha_tab[i] * 4 * np.pi * \
        #           quad(lambda x: x ** 2. * 3.8718 * 10 ** -5. *
        #                          np.exp(- 2. / 0.2 * ((x / 113.46) ** 0.2 - 1.)), 0., 300.)[0]
        plt_info = beta_tab[i] * mass_list ** alpha_tab[i] * 4 * np.pi * 200. ** 3. / 3.
        print plt_info
        plt.plot(mass_list, plt_info, color=color_tab[i])

    fname = 'Uncertainty_Subhalo_Mass_Density.pdf'
    pl.savefig(dir + '../../../Plots/' + fname)
    return


def limit_comparison_varydNdM(bmin=20., annih_prod='BB', nobs=False):
    ptag = 'Pointlike'
    if nobs:
        ntag = ''
    else:
        ntag = '_Nobs_False_'
    dir = MAIN_PATH + '/SubhaloDetection/Data/'

    hw_3 = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000__Nobs_True_.dat')
    hw_1 = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                      annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                      '_Mlow_5.000_m18_Nobs_True_.dat')

    hw_5 = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                      annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                      '_Mlow_5.000_m20_Nobs_True_.dat')


    mass_list = np.logspace(1., 3., 100)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** 1., 10. ** 3.])
    pl.ylim([10 ** -27., 10. ** -23.])

    pts3 = interpola(mass_list, hw_3[:,0], hw_3[:, 1])
    plt.plot(mass_list, pts3, lw=1, color='k', alpha=1)

    pts1 = interpola(mass_list, hw_1[:, 0], hw_1[:, 1])
    pts5 = interpola(mass_list, hw_5[:, 0], hw_5[:, 1])
    plt.fill_between(mass_list, pts1, pts5, where=pts5 <= pts1, facecolor='blue',
                    edgecolor='None', interpolate=True, alpha=0.3)

    ltop = 10 ** -23.4
    ldwn = 10 ** -.3
    plt.text(15, ltop, r'$\chi \chi$ ' + r'$\rightarrow$' + r' $b \bar{{b}}$', color='k', fontsize=16)
    plt.axhline(y=2.2 * 10 **-26., xmin=0, xmax=1, lw=1, ls='--', color='k', alpha=1)
    figname = dir + '../Plots/' + 'Uncertainty_Limit_Comparison_' + ptag +'annih_prod_' + \
              annih_prod + '_bmin_{:.0f}'.format(bmin) + ntag + '.pdf'
    plt.text(15, ltop * ldwn ** 2., r'$M_{{min}} = 10^{{5}} M_\odot$', color='k', fontsize=16)
    pl.xlabel(r'$m_\chi$   [GeV]', fontsize=20)
    pl.ylabel(r'$\left< \sigma v \right>$   [$\textrm{cm}^3 \textrm{s}^{{-1}}$]', fontsize=20)
    fig.set_tight_layout(True)
    pl.savefig(figname)
    return


