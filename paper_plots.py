import numpy as np
from subhalo import *
import os
import pickle
from scipy.interpolate import interp1d, interp2d,RectBivariateSpline
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
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16


def profile_comparison(plot_hw=True):
    minrad = -2.
    mass_list = [10. ** 4., 10 ** 5., 10 ** 6., 10 ** 7.]
    rb_list = np.logspace(-4, np.log10(1.), 20)
    gamma_list = np.linspace(0.2, 0.85 + 0.351 / 0.861 - 0.1, 20)
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
        r2_e_tr = np.logspace(np.log10(e_tr.max_radius), np.log10(e_tr.virial_radius), 100)
        r_n_ntr = np.logspace(minrad, np.log10(n_ntr.max_radius), 100)
        r2_n_ntr = np.logspace(np.log10(n_ntr.max_radius), np.log10(n_ntr.virial_radius), 20)

        den_e_tr1 = e_tr.density(r1_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_e_tr2 = e_tr.density(r2_e_tr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr = n_ntr.density(r_n_ntr) * GeVtoSolarM * kpctocm ** 3.
        den_n_ntr2 = n_ntr.density(r2_n_ntr) * GeVtoSolarM * kpctocm ** 3.


        ax[i%2, ii].set_xscale("log")
        ax[i%2, ii].set_yscale('log')
        ax[i%2, ii].set_xlim([10 ** minrad, 2. * 10. ** 1.])
        ax[i%2, ii].plot(r1_e_tr, den_e_tr1, lw=2, color='Black')
        ax[i%2, ii].plot(r2_e_tr, den_e_tr2, '--', lw=2, ms=1, color='Black')
        ax[i%2, ii].plot(r_n_ntr, den_n_ntr, lw=2, color='Magenta')
        ax[i%2, ii].plot(r2_n_ntr, den_n_ntr2, '--', ms=1, lw=2, color='Magenta')
        ax[i%2, ii].plot(r_hw, hw_marg, lw=2, color='Blue')
        ax[i%2, ii].text(2 * 10**-2, 10**10, r'Subhalo Mass: {:.1e} $M_\odot$'.format(m_sub), fontsize=10)
        ax[i%2, ii].set_yticklabels(['', '', '', r'$10^4$', '', r'$10^6$', '', r'$10^8$', '', r'$10^{10}$', ''])

        if i % 2 == 1:
            ax[i % 2, ii].set_xlabel('Radius [kpc]', fontsize=16)
            if i == 1:
                ax[i % 2, ii].set_xticklabels(['', r'$10^{-2}$', r'$10^{-1}$', '1', ''])
            else:
                ax[i % 2, ii].set_xticklabels(['', '', r'$10^{-1}$', '1', r'$10$'])
        if ii == 0:
            ax[i%2, ii].set_ylabel(r'$\rho [M_\odot / kpc^3$]', fontsize=16)
            if i == 0:
                ax[i%2, ii].text(.5, 10**8.5, 'Bertoni et al.', fontsize=10, color='k')
                ax[i%2, ii].text(.5, 10 ** 7.8, 'Schoonenberg et al.', fontsize=10, color='Magenta')
                ax[i%2, ii].text(.5, 10 ** 7., 'This Work', fontsize=10, color='blue')
        ax[i%2, ii].set_ylim([10. ** 3., 10. ** 11.])
    pl.suptitle('Profile Comparison', fontsize=18)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Multi_plot_Density_Comparison_msubhalo_{:.2e}'.format(m_sub) + \
        '_BLH_Bertone_HW.pdf'
    plt.subplots_adjust(wspace=0.0, hspace=0.)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def subhalo_comparison():
    rb_list = np.logspace(-3, np.log10(1.), 20)
    gamma_list = np.linspace(0.2, 0.85 + 0.351 / 0.861 - 0.1, 20)
    #via_lac = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ViaLacteaII_Useable_Subhalos.dat')
    elvis = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ELVIS_Useable_Subhalos.dat')
    vl_num = [19557, 19643]
    elv_num = [20580, 8384]
    sub_list = []
    for i in vl_num:
        sub_list.append(via_lactea[i])
    for i in elv_num:
        sub_list.append(elvis[i])

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
        ax[i%2, ii].text(9., 2*10**4, r'Subhalo Mass: {:.1e} $M_\odot$'.format(m_sub),
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
    pl.suptitle('Profile Comparison', fontsize=18)
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    fig_name = folder + '../Plots/' + 'Sim_Comparison_msubhalo_BLH_Bertone_HW.pdf'
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


def limit_comparison(plike=True, bmin=20., annih_prod='BB'):
    if plike:
        ptag = 'Pointlike'
    else:
        ptag = 'Extended'
    dir = MAIN_PATH + '/SubhaloDetection/Data/'
    nfw = np.loadtxt(dir + 'Limits_' + ptag + '_NFW_Truncate_False_alpha_0.16_annih_prod_' +
                     annih_prod + '_arxiv_num_160106781_bmin_{:.1f}'.format(bmin) + '.dat')
    blh_old = np.loadtxt(dir + 'Limits_' + ptag + '_Einasto_Truncate_True_alpha_0.16_annih_prod_' +
                     annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) + '_Mlow_0.000_OLD.dat')
    blh_new = np.loadtxt(dir + 'Limits_' + ptag + '_Einasto_Truncate_True_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) + '_Mlow_0.000_NEW.dat')
    hw_low = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                        '_Mlow_-5.000_' + '.dat')
    hw_high = np.loadtxt(dir + 'Limits_' + ptag + '_HW_Truncate_False_alpha_0.16_annih_prod_' +
                         annih_prod + '_arxiv_num_13131729_bmin_{:.1f}'.format(bmin) +
                         '_Mlow_5.000_' + '.dat')

    list = [hw_low, hw_high, nfw, blh_old, blh_new]
    color_list = ['Blue', 'Blue', 'Magenta', 'Black', 'Black']
    ls_list = ['--', '--', '--', '--', '-.']
    mass_list = np.logspace(1., 3., 100)

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale('log')
    pl.xlim([10 ** 1., 10. ** 3.])
    pl.ylim([10 ** -27., 10. ** -23.])
    for i, lim in enumerate(list):
        pts = interpola(mass_list, lim[:, 0], lim[:, 1])
        plt.plot(mass_list, pts, ls=ls_list[i], lw=1, color=color_list[i])

    lowm = interpola(mass_list, hw_low[:, 0], hw_low[:, 1])
    highm = interpola(mass_list, hw_high[:, 0], hw_high[:, 1])
    ax.fill_between(mass_list, lowm, highm, where=highm >= lowm,
                    facecolor='Blue', edgecolor='None', interpolate=True, alpha=0.3)

    ltop = 10 ** -24.2
    ldwn = 10 ** -.4
    plt.text(15, ltop, 'NFW (Post-tidal Stripping)', color='Magenta', fontsize=10)
    plt.text(15, ltop * ldwn, 'Einasto (Pre-tidal Stripping)', color='k', fontsize=10)
    plt.text(15, ltop * ldwn ** 2., 'This Work', color='blue', fontsize=10)

    figname = dir + 'Limit_Comparison_' + ptag +'annih_prod_' + \
              annih_prod + '_bmin_{:.1f}'.format(bmin) + '.pdf'

    pl.xlabel(r'$m_\chi$   [GeV]', fontsize=20)
    pl.ylabel(r'$\left< \sigma v \right>$   [$cm^3 s^{{-1}}$]', fontsize=20)
    fig.set_tight_layout(True)
    pl.savefig(figname)
    return


def hw_prob_rb(rb, mass):
    rb_norm = 10. ** (-4.24) * mass ** 0.459
    sigma_c = 0.47
    return (np.exp(- (np.log(rb / rb_norm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
            (np.sqrt(2. * np.pi) * sigma_c * rb))


def hw_prob_gamma(gam):
    sigma = 0.426
    norm = 0.9
    k = 0.1
    mu = 0.85
    y = -1. / k * np.log(1. - k * (gam - mu) / sigma)
    return np.exp(- y ** 2. / 2.) / (np.sqrt(2. * np.pi) * (sigma - k * (gam - mu))) / norm


def obtain_number_density():
    max_mass = 10. ** 10.
    min_mass = 10. ** 8.
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
    subhalos = np.vstack((file_iso[:, :10], file_hi[:, :10]))
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
    mass_bins = np.logspace(np.log10(np.min(subhalos[:, 9])), 10.,
                            (10. - np.log10(np.min(subhalos[:, 9]))) * 15)
    fit_ein, bine = np.histogram(subhalos[:, 9], bins=mass_bins, normed=False, weights=1. / subhalos[:, 9])
    d_ein = np.zeros(len(mass_bins) - 1)

    for i in range(len(mass_bins) - 1):
            d_ein[i] = np.median(subhalos[(subhalos[:, 9] < mass_bins[i + 1]) &
                                          (subhalos[:, 9] > mass_bins[i])][:, 9])

    parms, cov = curve_fit(lin_fit, np.log10(d_ein), np.log10(fit_ein))
    plot_ein = lin_fit(np.log10(mass_bins), parms[0], parms[1])
    plt.plot(mass_bins, 10. ** plot_ein, color='red', lw=2)
    plt.text(2 * 10 ** 8., 10. ** -4.5, 'Fit Range: [{:.1e}, {:.1e}] $M_\odot$'.format(min_mass, max_mass),
             fontsize=14, color='r')
    plt.text(7. * 10 ** 6., 10. ** -7., 'ELVIS Subhalos: \n  $r \leq 300$ kpc'.format(min_mass, max_mass),
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
    plt.ylabel(r'$\frac{dN}{dV}$', fontsize=18)

    fit_ein, bine = np.histogram(subhalos[:, 1], bins=dist_bins, normed=False,
                                 weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.))
    d_ein = np.zeros(len(dist_bins) - 1)
    for i in range(len(dist_bins) - 1):
        d_ein[i] = np.median(subhalos[(subhalos[:, 1] < dist_bins[i + 1]) &
                                      (subhalos[:, 1] > dist_bins[i])][:, 1])

    parms, cov = curve_fit(einasto_fit, d_ein, fit_ein, bounds=([30., 0.2, 0.], [300., .7, .1]), sigma=fit_ein)
    lden = einasto_fit(8.5, parms[0], parms[1], parms[2])
    print 'Local: ', lden
    hist_norm = 0.9 * lden / (n_halos * (min_mass ** (-0.9) - max_mass ** (-0.9)))
    plot_ein = einasto_fit(dist_bins, parms[0], parms[1], parms[2])
    plt.plot(dist_bins, plot_ein, color='red', lw=2)
    plt.text(12., 10. ** -2.3, 'Fit Range: [{:.1e}, {:.1e}] $M_\odot$'.format(min_mass, max_mass),
             fontsize=14, color='k')

    plt.text(100., 10. ** -2.4, 'Einasto Parameters:', fontsize=14, color='b')
    plt.text(100., 10. ** -2.7, r'$\alpha = {:.2f}$'.format(parms[1]), fontsize=14, color='b')
    plt.text(100., 10. ** -3., r'$r_s = {:.2f}$'.format(parms[0]), fontsize=14, color='b')
    plt.text(12., 10. ** -4.5, r'$\frac{{d N}}{{dM dV}} = \frac{{{:.0f}}}{{kpc^{{3}}}} $'.format(hist_norm) +
             r'$\left(\frac{M}{M_\odot}\right)^{-1.9}$', fontsize=14, color='k')

    fname = 'NumberDensity_Histogram_ELVIS.pdf'
    pl.savefig(dir + '../../../Plots/' + fname)

    return