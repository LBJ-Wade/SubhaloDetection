# -*- coding: utf-8 -*-
"""
Created on Thur Aug 18 2016

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


class ELVIS(object):

    def __init__(self, profile=0, alpha=0.16, c=None, truncate=False, arxiv_num=13131729):
        self.dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ELVIS_Halo_Catalogs/'
        self.dir_hi_res = 'HiResCatalogs/'
        self.dir_iso = 'IsolatedCatalogs/'
        self.dir_paired = 'PairedCatalogs/'
        self.f_name = 'Elvis_Useable_Subhalos.dat'

        self.profile = profile
        self.truncate = truncate
        self.alpha = alpha
        self.arxiv_num = arxiv_num
        profile_list = ['einasto', 'nfw']
        self.profile_name = profile_list[self.profile]

        self.c = c

    def compile_all_subhalos(self):
        hi_res = glob.glob(self.dir + self.dir_hi_res + '*')
        iso = glob.glob(self.dir + self.dir_iso + '*')
        paired = glob.glob(self.dir + self.dir_paired + '*')
        file_hi = 'Hi_Res_Subhalos.dat'
        file_iso = 'Iso_Subhalos.dat'
        file_paired = 'Paired_Subhalos.dat'
        for f in hi_res:
            load_sim = np.loadtxt(f)
            load_sim = load_sim[load_sim[:, 7] > 5.]
            try:
                l_old = np.loadtxt(self.dir + file_hi)
                np.savetxt(self.dir + file_hi, np.vstack((load_sim, l_old)))
            except IOError:
                np.savetxt(self.dir + file_hi, load_sim)
        for f in iso:
            load_sim = np.loadtxt(f)
            try:
                l_old = np.loadtxt(self.dir + file_iso)
                np.savetxt(self.dir + file_iso, np.vstack((load_sim, l_old)))
            except IOError:
                np.savetxt(self.dir + file_iso, load_sim)
        for f in paired:
            load_sim = np.loadtxt(f)
            try:
                l_old = np.loadtxt(self.dir + file_paired)
                np.savetxt(self.dir + file_paired, np.vstack((load_sim, l_old)))
            except IOError:
                np.savetxt(self.dir + file_paired, load_sim)
        return

    def kill_unuseable_halos(self):
        """
        FILE OUTPUT FORMAT:
        ID, GC Dist (kpc), Vmax (hist) (km/s), Vmax (z=0), Rmax (z=0),
        Mass (SM), R_tidal (kpc), 3-pos (MPC), 3-vel, Number of Particles,
        mass per particle, HAS_BEEN_NEAR_GC_RECENTLY (1/0)
        """
        load_fname_hi = self.dir + 'Hi_Res_Subhalos.dat'
        load_fname_iso = self.dir + 'Iso_Subhalos.dat'
        load_fname_pair = self.dir + 'Paired_Subhalos.dat'
        f_name = '../ELVIS_Useable_Subhalos.dat'
        hi_res = np.loadtxt(load_fname_hi)
        iso = np.loadtxt(load_fname_iso)
        pair = np.loadtxt(load_fname_pair)

        hi_neargc, iso_neargc, pair_neargc = self.evolution_tracks()

        h = 0.71
        gc_pos = np.array([25. / h, 25. / h, 25. / h])
        index_toss = []
        file_ord = np.zeros(len(iso) * 16).reshape((len(iso), 16))
        rvir = iso[0, 11]
        for i, f in enumerate(iso):
            if i == 0:
                index_toss.append(i)
            if f[0] in iso_neargc:
                hist_gc = 1
            else:
                hist_gc = 0
            gcd = np.sqrt(np.dot(gc_pos - f[1:4], gc_pos - f[1:4])) * 1000.
            file_ord[i] = [f[0], gcd, f[8], f[7], f[12], f[9], f[11], f[1], f[2], f[3],
                           f[4], f[5], f[6], f[16], 1.9 * 10. ** 5., hist_gc]
            if file_ord[i, 1] > rvir:
                index_toss.append(i)
            file_ord[i, 4] /= 1.045

        index_toss = list(set(index_toss))
        full_ind = set(range(len(iso)))
        ind_of_int = list(full_ind.difference(index_toss))
        iso = file_ord[ind_of_int]

        index_toss = []
        file_ord = np.zeros(len(hi_res) * 16).reshape((len(hi_res), 16))
        rvir = hi_res[0, 11]
        for i, f in enumerate(hi_res):
            if i == 0:
                index_toss.append(i)
            if f[0] in hi_neargc:
                hist_gc = 1
            else:
                hist_gc = 0
            gcd = np.sqrt(np.dot(gc_pos - f[1:4], gc_pos - f[1:4])) * 1000.
            file_ord[i] = [f[0], gcd, f[8], f[7], f[12], f[9], f[11], f[1], f[2], f[3],
                        f[4], f[5], f[6], f[16], 2.35 * 10. ** 4., hist_gc]
            if file_ord[i, 1] > rvir:
                index_toss.append(i)
            file_ord[i, 4] /= 1.045

        index_toss = list(set(index_toss))
        full_ind = set(range(len(hi_res)))
        ind_of_int = list(full_ind.difference(index_toss))
        hi_res = file_ord[ind_of_int]

        index_toss = []
        gc_pos1 = np.array([pair[0, 1], pair[0, 2], pair[0, 3]])
        gc_pos2 = np.array([pair[1, 1], pair[0, 2], pair[0, 3]])
        file_ord = np.zeros(len(pair) * 16).reshape((len(pair), 16))
        rvir = np.max([pair[0, 11], pair[1, 11]])
        for i, f in enumerate(pair):
            if i == 0 or i == 1:
                index_toss.append(i)
            if f[0] in pair_neargc:
                hist_gc = 1
            else:
                hist_gc = 0
            gcd_1 = np.sqrt(np.dot(gc_pos1 - f[1:4], gc_pos1 - f[1:4])) * 1000.
            gcd_2 = np.sqrt(np.dot(gc_pos2 - f[1:4], gc_pos2 - f[1:4])) * 1000.
            gcd = np.min([gcd_1, gcd_2])
            file_ord[i] = [f[0], gcd, f[8], f[7], f[12], f[9], f[11], f[1], f[2], f[3],
                           f[4], f[5], f[6], f[16], 1.9 * 10. ** 5., hist_gc]
            if file_ord[i, 1] > rvir:
                index_toss.append(i)
            file_ord[i, 4] /= 1.045

        index_toss = list(set(index_toss))
        full_ind = set(range(len(pair)))
        ind_of_int = list(full_ind.difference(index_toss))
        pair = file_ord[ind_of_int]

        elvis_file = np.vstack((iso, pair, hi_res))
        np.savetxt(self.dir + f_name, elvis_file, fmt='%.2f')
        return

    def find_subhalos(self, min_mass=4.1 * 10 ** 3., max_mass=10. ** 12., gcd_min=0., gcd_max=4000.,
                      del_vmax_min=0., del_vmax_max=1., print_info=True):
        elvis = np.loadtxt(self.dir + '../' + self.f_name)
        list_of_interest = []
        for i,f in enumerate(elvis):
            del_vmax = (f[2] - f[3]) / f[2]
            if (min_mass < f[5] < max_mass) and (gcd_min < f[1] < gcd_max) and \
                (del_vmax_min < del_vmax < del_vmax_max):
                list_of_interest.append(i)
        print 'Mass Range: ', [min_mass, max_mass]
        print 'GC Distance Range: ', [gcd_min, gcd_max]
        if print_info:
            print 'Subhalos Matching: ', list_of_interest
            return
        else:
            return list_of_interest

    def find_mean_mass_differential(self):
        elvis = np.loadtxt(self.dir + '../' + self.f_name)
        pair_iso_err = np.array([])
        hires_err = np.array([])
        for i, sub in enumerate(elvis):
            if sub[5] < 10. ** 8.:
                if sub[-2] == 1.9 * 10 ** 5.:
                    pair_iso_err = np.append(pair_iso_err,
                                             (sub[5] - sub[-3] * sub[-2]) / (sub[-3] * sub[-2]))
                elif sub[-2] == 2.35 * 10 ** 4.:
                    hires_err = np.append(pair_iso_err,
                                          (sub[5] - sub[-3] * sub[-2]) / (sub[-3] * sub[-2]))
        print 'Num Halos: ', pair_iso_err.size + hires_err.size

        print 'Pair/Iso Mass Err: ', np.mean(np.abs(pair_iso_err)), \
            np.median(np.abs(pair_iso_err)), np.sum(pair_iso_err) / pair_iso_err.size
        print 'Hi-res Mass Err: ', np.mean(np.abs(hires_err)), \
            np.median(np.abs(hires_err)), np.sum(hires_err) / hires_err.size

        return

    def evolution_tracks(self):

        hres = glob.glob(self.dir + 'HiResTrees/*')

        hres_recent_near_gc = []

        for ex in hres:
            x_hist = np.loadtxt(ex + '/X.txt')
            y_hist = np.loadtxt(ex + '/Y.txt')
            z_hist = np.loadtxt(ex + '/Z.txt')
            ids = np.loadtxt(ex + '/ID.txt')
            host_x = x_hist[0]
            host_y = y_hist[0]
            host_z = z_hist[0]

            dir_st = ex.find('HiResTrees')
            hires_or = np.loadtxt(ex[:dir_st] + 'HiResCatalogs/' + ex[dir_st+10:] + '.txt')

            for i, sub in enumerate(hires_or):
                if i != 0:
                    id_num = sub[0]
                    x, y, z = [x_hist[i], y_hist[i], z_hist[i]]
                    gcd = np.sqrt(10. ** 6. * ((x - host_x) ** 2. + (y - host_y) ** 2. + (z - host_z) ** 2.))

                    if np.min(gcd[:9]) < 20.:
                        hres_recent_near_gc.append(id_num)

        iso_recent_near_gc = []
        iso = glob.glob(self.dir + 'IsolatedTrees/*')

        for ex in iso:
            x_hist = np.loadtxt(ex + '/X.txt')
            y_hist = np.loadtxt(ex + '/Y.txt')
            z_hist = np.loadtxt(ex + '/Z.txt')
            ids = np.loadtxt(ex + '/ID.txt')
            host_x = x_hist[0]
            host_y = y_hist[0]
            host_z = z_hist[0]

            dir_st = ex.find('IsolatedTrees')
            iso_or = np.loadtxt(ex[:dir_st] + 'IsolatedCatalogs/' + ex[dir_st + 13:] + '.txt')

            for i, sub in enumerate(iso_or):
                if i != 0:
                    id_num = sub[0]
                    x, y, z = [x_hist[i], y_hist[i], z_hist[i]]
                    gcd = 10. ** 3. * np.sqrt(
                        ((x - host_x) ** 2. + (y - host_y) ** 2. + (z - host_z) ** 2.))

                    if np.min(gcd[:9]) < 20.:
                        iso_recent_near_gc.append(id_num)

        pair = glob.glob(self.dir + 'PairedTrees/*')
        pair_recent_near_gc = []

        for ex in pair:
            x_hist = np.loadtxt(ex + '/X.txt')
            y_hist = np.loadtxt(ex + '/Y.txt')
            z_hist = np.loadtxt(ex + '/Z.txt')
            ids = np.loadtxt(ex + '/ID.txt')
            host_x = x_hist[0]
            host_y = y_hist[0]
            host_z = z_hist[0]

            dir_st = ex.find('PairedTrees')
            pair_or = np.loadtxt(ex[:dir_st] + 'PairedCatalogs/' + ex[dir_st + 11:] + '.txt')

            for i, sub in enumerate(pair_or):
                if i != 0:
                    id_num = sub[0]
                    x, y, z = [x_hist[i], y_hist[i], z_hist[i]]
                    gcd = 10. ** 3. * np.sqrt(
                        ((x - host_x) ** 2. + (y - host_y) ** 2. + (z - host_z) ** 2.))

                    if np.min(gcd[:35]) < 20.:
                        pair_recent_near_gc.append(id_num)

        return hres_recent_near_gc, iso_recent_near_gc, pair_recent_near_gc

    def obtain_number_density(self, min_mass=1.0e+7, max_mass=1.e+10):

        gcd_range = np.array([0., 300.])
        sub_o_int = self.find_subhalos(min_mass=min_mass, max_mass=max_mass,
                                       gcd_min=gcd_range[0], gcd_max=gcd_range[-1],
                                       print_info=False)
        elvis = np.loadtxt(self.dir + '../' + self.f_name)
        subhalos = elvis[sub_o_int]
        print 'Num Subhalos: ', len(subhalos)
        n_halos = 24. + 24. + 3.
        file_hi = np.loadtxt(self.dir + 'Hi_Res_Subhalos.dat')
        file_iso = np.loadtxt(self.dir + 'Iso_Subhalos.dat')
        file_paired = np.loadtxt(self.dir + 'Paired_Subhalos.dat')

        full_subhalos = np.concatenate((file_hi[:, 9], file_iso[:, 9], file_paired[:, 9]))
        full_subhalos = full_subhalos[full_subhalos > min_mass]

        dist_bins = np.linspace(1., 300., 20)
        min_mass = np.min(subhalos[:, 5])
        max_mass = np.max(subhalos[:, 5])
        print 'Making Subhalo Number Density Histogram...'
        plt.hist(subhalos[:, 1], bins=dist_bins, log=True, normed=False,
                 weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.),
                 color='White')

        pl.gca().set_xscale("log")
        pl.xlim([10., 300])
        plt.xlabel(r'Distance', fontsize=16)
        plt.ylabel(r'$\frac{dN}{dV}$', fontsize=16)

        fit_ein, bine = np.histogram(subhalos[:, 1], bins=dist_bins, normed=False,
                                     weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.))
        d_ein = np.zeros(len(dist_bins) - 1)
        for i in range(len(dist_bins) - 1):
            d_ein[i] = np.median(subhalos[(subhalos[:, 1] < dist_bins[i + 1]) &
                                          (subhalos[:, 1] > dist_bins[i])][:, 1])


        parms, cov = curve_fit(einasto_fit, d_ein, fit_ein, bounds=(0, np.inf), sigma=fit_ein)
        lden = einasto_fit(8.5, parms[0], parms[1], parms[2])
        hist_norm = 0.9 * lden / (n_halos * (min_mass ** (-0.9) - max_mass ** (-0.9)))
        print lden, min_mass, max_mass, n_halos, hist_norm
        plot_ein = einasto_fit(dist_bins, parms[0], parms[1], parms[2])
        plt.plot(dist_bins, plot_ein, '--', color='blue')
        plt.text(11., 10. ** -1.4, 'Mass Range: [{:.2e}, {:.2e}]'.format(min_mass, max_mass),
                 fontsize=10, color='k')
        plt.text(100., 10. ** -2., 'Einasto Fit', fontsize=10, color='b')
        plt.text(100., 10. ** -2.2, r'$\alpha = {:.2f}$'.format(parms[1]), fontsize=10, color='b')
        plt.text(100., 10. ** -2.4, r'$r_s = {:.2f}$'.format(parms[0]), fontsize=10, color='b')
        plt.text(100., 10. ** -3., r'$\frac{{d N}}{{dM dV}} = \frac{{{:.0f}}}{{kpc^{{3}}}} $'.format(hist_norm) +
                                   r'$\left(\frac{M}{M_\odot}\right)^{-1.9}$', fontsize=14, color='k')

        fname = 'ELVIS_Subhalo_Number_Density_Mass_Range_{:.1e}_{:.1e}'.format(min_mass, max_mass) + \
                'GCD_range_{:.1f}_{:.1f}'.format(gcd_range[0], gcd_range[1]) + '.pdf'
        pl.savefig(self.dir + '../Elvis_plots/' + fname)

        return

    def obtain_number_density_paired(self, min_mass=1.0e+8, max_mass=1.e+10):

        gcd_range = np.array([0., 300.])
        n_halos = 24.
        file_paired = np.loadtxt(self.dir + 'Paired_Subhalos.dat')
        ind_of_int = []
        gcd1 = file_paired[0, 1:4]
        gcd2 = file_paired[1, 1:4]

        for i, sub in enumerate(file_paired):
            if i == 0 or i == 1:
                pass
            else:
                dist1 = np.sqrt(((gcd1 - sub[1:4]) ** 2.).sum())
                dist2 = np.sqrt(((gcd2 - sub[1:4]) ** 2.).sum())
                dist = np.min([dist1, dist2]) * 10**3.
                sub[1] = dist
                if dist < 300.:
                    ind_of_int.append(i)

        subhalos = file_paired[ind_of_int]
        min_mass = np.min(subhalos[:, 9])
        max_mass = np.max(subhalos[:, 9])
        print 'Num Subhalos: ', len(subhalos)

        dist_bins = np.linspace(1., 300., 20)
        print 'Making Subhalo Number Density Histogram...'
        plt.hist(subhalos[:, 1], bins=dist_bins, log=True, normed=False,
                 weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.),
                 color='White')

        pl.gca().set_xscale("log")
        pl.xlim([10., 300])
        pl.ylim([10.**-6., 10.**-3.])
        plt.xlabel(r'Distance', fontsize=16)
        plt.ylabel(r'$\frac{dN}{dV}$', fontsize=16)

        fit_ein, bine = np.histogram(subhalos[:, 1], bins=dist_bins, normed=False,
                                     weights=1. / (4. / 3. * np.pi * subhalos[:, 1] ** 3.))
        d_ein = np.zeros(len(dist_bins) - 1)
        for i in range(len(dist_bins) - 1):
            d_ein[i] = np.median(subhalos[(subhalos[:, 1] < dist_bins[i + 1]) &
                                          (subhalos[:, 1] > dist_bins[i])][:, 1])
        print d_ein, fit_ein
        parms, cov = curve_fit(einasto_fit, d_ein, fit_ein, bounds=([30., 0.2, 0.], [300., .7, .1]),sigma=fit_ein)
        lden = einasto_fit(8.5, parms[0], parms[1], parms[2])
        hist_norm = 0.9 * lden / (n_halos * (min_mass ** (-0.9) - max_mass ** (-0.9)))
        plot_ein = einasto_fit(dist_bins, parms[0], parms[1], parms[2])
        plt.plot(dist_bins, plot_ein, '--', color='blue')
        plt.text(11., 10. ** -3.2, 'Mass Range: [{:.2e}, {:.2e}]'.format(min_mass, max_mass),
                 fontsize=10, color='k')
        plt.text(100., 10. ** -3.2, 'PARIED', fontsize=10, color='red')
        plt.text(100., 10. ** -3.4, 'Einasto Fit', fontsize=10, color='b')
        plt.text(100., 10. ** -3.6, r'$\alpha = {:.2f}$'.format(parms[1]), fontsize=10, color='b')
        plt.text(100., 10. ** -3.8, r'$r_s = {:.2f}$'.format(parms[0]), fontsize=10, color='b')
        plt.text(100., 10. ** -4.1, r'$\frac{{d N}}{{dM dV}} = \frac{{{:.2f}}}{{kpc^{{3}}}} $'.format(hist_norm) +
                                    r'$\left(\frac{M}{M_\odot}\right)^{-1.9}$', fontsize=14, color='k')

        fname = 'ELVIS_PARIED_Subhalo_Number_Density_Mass_Range_{:.1e}_{:.1e}'.format(min_mass, max_mass) + \
                'GCD_range_{:.1f}_{:.1f}'.format(gcd_range[0], gcd_range[1]) + '.pdf'
        pl.savefig(self.dir + '../Elvis_plots/' + fname)
        return

def plot_sample_comparison_elv(sub_num=0, plot=True, show_plot=False):

    color_list = ['Aqua', 'Magenta', 'Orange', 'Green', 'Red', 'Brown', 'Purple']
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ELVIS_Useable_Subhalos.dat'
    elvis = np.loadtxt(dir + f_name)
    # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
    #                     rel_pos, rel_pos, Npart]                         (7-14)

    s_mass = elvis[sub_num, 5]
    subhalo_e_t = Einasto(s_mass / 0.005, 0.16, truncate=True, arxiv_num=13131729)
    subhalo_n_bert = NFW(s_mass, 0.16, truncate=False, arxiv_num=160106781,
                         vmax=elvis[sub_num, 3], rmax=elvis[sub_num, 4])
    try:
        bf_alpha = find_alpha(s_mass, elvis[sub_num, 3], elvis[sub_num, 4])
    except ValueError:
        print 'Setting alpha = 1 for Einasto profile'
        bf_alpha = 1.

    try:
        bf_gamma = find_gamma(s_mass, elvis[sub_num, 3], elvis[sub_num, 4],
                              error=np.sqrt(elvis[sub_num, -3]) * elvis[sub_num, -2])
    except ValueError:
        print 'Setting gamma = 1 for KMMDSM profile'
        bf_gamma = .0
    bf_line = find_r1_r2(s_mass, elvis[sub_num, 3], elvis[sub_num, 4])
    sub_gen_n = []
    r1_list = []
    r2_list = []
    vmax_load = elvis[sub_num, 3]
    rmax_load = elvis[sub_num, 4]

    print 'Halo Rvmax, Vmax: ', rmax_load, vmax_load, elvis[sub_num, 5]

    subhalo_e_fit = Einasto(s_mass, alpha=bf_alpha, arxiv_num=160106781,
                            vmax=vmax_load, rmax=rmax_load)
    subhalo_kmmdsm = KMMDSM(s_mass, bf_gamma, arxiv_num=160106781,
                            vmax=vmax_load, rmax=rmax_load)
    print bf_gamma, subhalo_kmmdsm.rb
    for i in range(len(bf_line[:, 0])):
        r1, r2 = bf_line[i]
        r1_list.append(r1)
        r2_list.append(r2)
        sub_gen_n.append(Over_Gen_NFW(s_mass, r1, r2, vmax=vmax_load, rmax=rmax_load))
        print 'r1, r2: ', r1, r2
        print 'Scale Radius: ', sub_gen_n[i].scale_radius
    r1_list.sort()
    r2_list.sort()

    num_d_pts = 100
    r_tab = np.logspace(-2., 1., num_d_pts)

    den_e_t = np.zeros_like(r_tab)
    den_n_bert = np.zeros_like(r_tab)
    den_e_fit = np.zeros_like(r_tab)
    den_kmmdsm_fit = np.zeros_like(r_tab)
    den_n_gen_n = np.zeros(r_tab.size * len(sub_gen_n)).reshape((len(sub_gen_n), r_tab.size))
    for i in range(num_d_pts):
        den_e_t[i] = subhalo_e_t.int_over_density(r_tab[i])
        den_n_bert[i] = subhalo_n_bert.int_over_density(r_tab[i])
        den_e_fit[i] = subhalo_e_fit.int_over_density(r_tab[i])
        den_kmmdsm_fit[i] = subhalo_kmmdsm.int_over_density(r_tab[i])
        for j in range(len(sub_gen_n)):
            den_n_gen_n[j, i] = sub_gen_n[j].int_over_density(r_tab[i])

    sub = elvis[sub_num]
    del_m_list = np.array([[0., 0., 0.],
                           [sub[4], sub[3] ** 2. * sub[4] / newton_G,
                            np.sqrt(sub[3] ** 2. * sub[4] * sub[-2] / newton_G)],
                           [sub[6], sub[5], np.sqrt(sub[-3] * sub[-2])]])

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
                yerr_h = (np.log10(del_m_list[x + 1, 1] + 3. * del_m_list[x + 1, 2]) -
                              np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin))
            except:
                yerr_l = 0.
                yerr_h = 1.
            print 'Err1: ', 10. ** (yerr_l * (np.log10(ymax) - np.log10(ymin)) + np.log10(ymin))
            print 'Err2: ', del_m_list[x + 1, 1] - 3. * del_m_list[x + 1, 2]
            print 'Mass', del_m_list[x+1, 1]
            print 'Delta M ', del_m_list[x + 1, 2]

            plt.axvline(x=del_m_list[x + 1, 0], ymin=yerr_l, ymax=yerr_h, linewidth=1, color='Black')

        plt.plot(r_tab, den_e_t, lw=1, color=color_list[0], label='Ein (T)')
        plt.plot(r_tab, den_n_bert, lw=1, color=color_list[5], label='NFW (Bertone)')
        plt.plot(r_tab, den_e_fit, lw=1, color=color_list[3], label='Einasto (Fit)')
        plt.plot(r_tab, den_kmmdsm_fit, lw=1, color=color_list[2], label='KMMDSM')
        for i in range(len(sub_gen_n)):
            plt.plot(r_tab, den_n_gen_n[i], lw=1, color=color_list[1], label='DGen NFW', alpha=0.2)

        plt.text(2., 1. * 10 ** 4.6, 'KMMDSM', fontsize=10, ha='left', va='center', color=color_list[2])
        plt.text(2., 1. * 10 ** 4.3, r'$\gamma =$ ' + '{:.2f}'.format(bf_gamma), fontsize=10,
                 ha='left', va='center', color=color_list[2])
        plt.text(2., 1. * 10 ** 4., 'Einasto (T)', fontsize=10, ha='left', va='center', color=color_list[0])
        plt.text(2., 1. * 10 ** 3.7, 'Einasto (Fit)', fontsize=10, ha='left', va='center', color=color_list[3])
        plt.text(2., 1. * 10 ** 3.4, r'$\alpha =$ ' + '{:.2f}'.format(bf_alpha),
                 fontsize=10, ha='left', va='center', color=color_list[3])
        plt.text(2., 1. * 10 ** 3.1, 'NFW (Bertone)', fontsize=10, ha='left', va='center', color=color_list[5])
        plt.text(2., 1. * 10 ** 2.8, 'DGen NFW', fontsize=10, ha='left', va='center', color=color_list[1])
        try:
            plt.text(2., 1. * 10 ** 2.5, r'$r_{inner}$ ' + '[{:.2f}, {:.2f}]'.format(r1_list[0], r1_list[-1]),
                     fontsize=10, ha='left', va='center', color=color_list[1])
            plt.text(2., 1. * 10 ** 2.2, r'$r_{outter}$ ' + '[{:.2f}, {:.2f}]'.format(r2_list[0], r2_list[-1]),
                     fontsize=10, ha='left', va='center', color=color_list[1])
        except:
            pass

        plt.text(1.3 * 10**-2., 1. * 10 ** 7., r'Halo Mass: {:.1e} $M_\odot$'.format(elvis[sub_num, 5]),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(1.3 * 10 ** -2., 1. * 10 ** 6.8, 'GC Dist {:.1f} kpc'.format(elvis[sub_num, 1]),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(1.3 * 10 ** -2., 1. * 10 ** 6.6, 'Vmax {:.1f} km/s'.format(elvis[sub_num, 3]),
                 fontsize=10, ha='left', va='center', color='Black')

        fig_name = dir + '/Elvis_plots/' + 'Subhalo_Number_' + str(sub_num) +\
            '_Profile_Comparison.pdf'

        pl.xlabel('Distance [kpc]', fontsize=20)
        pl.ylabel(r'M(r)  [$M_\odot$]', fontsize=20)
        fig.set_tight_layout(True)
        pl.savefig(fig_name)
        return
    else:
        return del_m_list, r_tab, den_e_t, den_n_bert


def multi_slice_elv(m_low=4.1 * 10 ** 3., m_high =10.**12., m_num=10, gc_d_min=0., gc_d_max=4000.,
                gc_d_num=2, plot=True, alpha=.1, ms=1, p_mtidal=False, p_mrmax=False):
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    color_list = ["Aqua", "Red", "Green", "Magenta", "Black"]
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
    for g in range(len(gcd_list) - 1):
        for m in range(m_list.size - 1):
            print [m_list[m], m_list[m+1]],[gcd_list[g],gcd_list[g+1]]
            collect_info = plot_slice_elv(m_low=m_list[m], m_high =m_list[m + 1], gc_d_min=gcd_list[g],
                                      gc_d_max=gcd_list[g + 1], plot=False, alpha=.2, ms=2)
            for sub in collect_info:
                if p_mrmax:
                    plt.plot(sub[0], sub[1], 'o', ms=ms, color=color_list[m + g], alpha=alpha)
                if p_mtidal:
                    plt.plot(sub[2], sub[3], 'o', ms=ms, color=color_list[m + g], alpha=alpha)

    plot_lab = ''
    if p_mtidal:
        plot_lab += '_plot_mtidal'
    if p_mrmax:
        plot_lab += '_plot_mrmax'

    fig_name = dir + '/Elvis_plots/' + 'Subhalo_MassDistSplice_with_Mmin_{:.2e}'.format(m_low) + \
               '_Mhigh_{:.2e}'.format(m_high) + '_M_num_' + str(m_num) +\
               '_GCd_low_{:.1f}'.format(gc_d_min) + '_GCd_high_{:.1f}'.format(gc_d_max) +\
               '_GCd_num_' + str(gc_d_num) + plot_lab + '.pdf'

    pl.xlabel('Distance [kpc]', fontsize=20)
    pl.ylabel(r'M(r)  [$M_\odot$]', fontsize=20)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)
    return


def plot_slice_elv(m_low=4.1 * 10 ** 3., m_high =10.**12., gc_d_min=0., gc_d_max=200., plot=True,
               alpha=.1, ms=1):
    color_list = ['Aqua', 'Magenta', 'Orange', 'Green', 'Red', 'Brown']
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ELVIS_Useable_Subhalos.dat'
    elvis_file = np.loadtxt(dir + f_name)
    # File Organization: [id, GC dist, peakVmax, Vmax, Rvmax, Mass, rtidal,     (0-6)
    #                     rel_pos, rel_pos, M300, M600]                         (7-14)
    collection = np.array([])
    for i,sub in enumerate(elvis_file):
        if m_low <= sub[5] <= m_high and gc_d_min <= sub[1] <= gc_d_max:
            try:
                collection = np.vstack((collection,
                                        np.array([sub[4], sub[3] ** 2. * sub[4] / newton_G,
                                                  sub[6], sub[5]])))
            except ValueError:
                collection = np.array([sub[4], sub[3] ** 2. * sub[4] / newton_G,
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
                labels = [r'$M_{R_{max}}$', r'$M_{tidal}$']
            else:
                labels = [None, None, None]
            plt.plot(h[0], h[1], 'o', ms=ms, color='Red', alpha = alpha, label=labels[0])
            plt.plot(h[2], h[3], 'o', ms=ms, color='Blue', alpha = alpha, label=labels[1])

        subhalo_e_t_1 = Einasto(min_m / 0.005, 0.16, truncate=True, arxiv_num=13131729)
        subhalo_n_bert_1 = NFW(min_m, 0.16, truncate=False, arxiv_num=160106781)
        subhalo_e_t_2 = Einasto(max_m / 0.005, 0.16, truncate=True, arxiv_num=13131729)
        subhalo_n_bert_2 = NFW(max_m, 0.16, truncate=False, arxiv_num=160106781)
        num_d_pts = 100
        r_tab = np.logspace(-2., 1., num_d_pts)
        den_e_t = np.zeros_like(r_tab)
        den_n_bert = np.zeros_like(r_tab)
        den_e_t2 = np.zeros_like(r_tab)
        den_n_bert2 = np.zeros_like(r_tab)
        for i in range(num_d_pts):
            den_e_t[i] = subhalo_e_t_1.int_over_density(r_tab[i])
            den_n_bert[i] = subhalo_n_bert_1.int_over_density(r_tab[i])
            den_e_t2[i] = subhalo_e_t_2.int_over_density(r_tab[i])
            den_n_bert2[i] = subhalo_n_bert_2.int_over_density(r_tab[i])
        plt.plot(r_tab, den_e_t, '-.', r_tab, den_e_t2, '-.', lw=1, color=color_list[0], label='Ein (T)', alpha=0.5)
        plt.plot(r_tab, den_n_bert, '-.', r_tab, den_n_bert, '-.', lw=1, color=color_list[5], label='NFW (Bertone)', alpha=0.5)
        plt.text(1.3 * 10 ** -2., 1. * 10 ** 8.7,
                 r'Halo Masses: [{:.1e}, {:.1e}] $M_\odot$'.format(m_low, m_high),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.text(1.3 * 10 ** -2., 1. * 10 ** 8.4,
                 r'GC Dist: [{:.1e}, {:.1e}] kpc'.format(gc_d_min, gc_d_max),
                 fontsize=10, ha='left', va='center', color='Black')
        plt.legend(loc=4, fontsize=10)
        fig_name = dir + '/Elvis_plots/' + 'Subhalos_with_Mmin_{:.2e}'.format(m_low) +\
            '_Mhigh_{:.2e}'.format(m_high) + '_GCd_low_{:.1f}'.format(gc_d_min) +\
            '_GCd_high_{:.1f}'.format(gc_d_max) + '.pdf'

        pl.xlabel('Distance [kpc]', fontsize=20)
        pl.ylabel(r'M(r)  [$M_\odot$]', fontsize=20)
        fig.set_tight_layout(True)
        pl.savefig(fig_name)
    else:
        return collection


def Preferred_Density_Slopes_elv(mass_low=10. ** 4, mass_high = 10.**8., gcd_min=0., gcd_max=200.,
                             tag=''):
    dir = MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/'
    f_name = 'ELVIS_Useable_Subhalos.dat'
    via_lac = np.loadtxt(dir + f_name)
    failures = 0
    soi = ELVIS().find_subhalos(min_mass=mass_low, max_mass=mass_high,
                                gcd_min=gcd_min, gcd_max=gcd_max, print_info=False)
    print 'Num Subhalos: ', len(soi)
    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    rtab = np.linspace(0., 2., 200)
    for i, halo in enumerate(via_lac[soi]):
        print i
        rmax, vmax, mass = [halo[4], halo[3], halo[5]]
        print mass, rmax, vmax

        try:
            bf_line = find_r1_r2(mass, vmax, rmax)
            plot_bf = interpola(rtab[rtab < bf_line[-1, 0]], bf_line[:, 0], bf_line[:, 1])
            pl.plot(rtab[rtab < bf_line[-1, 0]], plot_bf)
        except:
            failures += 1
            print 'Fail.'
            print 'Subhalo Number: ', soi[i]
    pl.xlim([0., 2.])
    plt.xlabel(r'$r_{inner}$', fontsize=16)
    plt.ylabel(r'$r_{outer}$', fontsize=16)
    plt.text(1.5, 40., 'Failures: {}'.format(failures),
             fontsize=10, ha='left', va='center', color='Black')
    plt.plot(1., 2., '*', ms=10)
    sv_fname = dir + '/Elvis_plots/InnerOuterSlope' + tag + '.pdf'
    print sv_fname
    print 'Failures: ', failures
    pl.savefig(sv_fname)
    return