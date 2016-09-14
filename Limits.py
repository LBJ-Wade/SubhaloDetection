# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

@author: SamWitte
"""
import numpy as np
import os
from math import factorial
from scipy.integrate import quad
from scipy.optimize import fminbound
from scipy.interpolate import interp1d
from scipy.special import gamma
import glob
from subhalo import *

try:
    MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'


class DM_Limits(object):
    """
    Given precalculated cross section vs n_obs files, this class calculates the
    the upper limits at a chosen CL
    """
    def __init__(self, nobs=True, nbkg=0., CL=0.9, annih_prod='BB', pointlike=True,
                 alpha=0.16, profile=0, truncate=False, arxiv_num=10070438, b_min=30.,
                 method=0, stiff_rb=False, m_low=np.log10(10**4.), tag='_'):

        Profile_list = ["Einasto", "NFW", "HW"]
        Method_list = ["Poisson", "FeldmanCousins"]
        self.method = Method_list[method]
        self.nobs = nobs
        self.nbkg = nbkg
        self.CL = CL
        self.annih_prod = annih_prod
        self.alpha = alpha
        self.profile = profile
        self.profile_name = Profile_list[self.profile]
        self.truncate = truncate
        self.arxiv_num = arxiv_num
        self.b_min = b_min
        self.pointlike = pointlike
        self.stiff_rb = stiff_rb
        self.m_low = m_low
        self.tag = tag
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"

    def poisson_limit(self):
        sources = np.loadtxt(self.folder + 'Misc_Items/WeightedSources_' + self.annih_prod + '.dat')
        if self.nobs:
            nobstag = '_Nobs_True_'
        else:
            nobstag = '_Nobs_False_'
        if self.pointlike:
            plike_tag = '_Pointlike'
        else:
            plike_tag = '_Extended'

        file_name = 'Limits' + plike_tag + '_' + self.profile_name + '_Truncate_' + \
            str(self.truncate) + '_alpha_' + str(self.alpha) +\
            '_annih_prod_' + self.annih_prod + '_arxiv_num_' +\
            str(self.arxiv_num) + '_bmin_' + str(self.b_min) +\
            '_Mlow_{:.3f}'.format(self.m_low) + self.tag + nobstag + '.dat'

        if self.profile < 2:
            extra_tag = '_Truncate_' + str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + \
                        '_alpha_' + str(self.alpha)
        else:
            extra_tag = '_Gamma_0.850_Stiff_rb_'+ str(self.stiff_rb)

        f_names = self.profile_name + '_mx_*' + '_annih_prod_' + self.annih_prod + '_bmin_' +\
               str(self.b_min) + plike_tag + extra_tag + '_Mlow_{:.3f}'.format(self.m_low) +\
               self.tag + '.dat'
        print f_names

        foi = glob.glob(self.folder + '/Cross_v_Nobs/' + f_names)
        limarr = np.zeros(2 * len(foi)).reshape((len(foi), 2))

        for i, f in enumerate(foi):
            print f
            mstart = f.find('_mx_')
            mstart += 4
            j = 0
            found_mass = False
            while not found_mass:
                try:
                    mx = float(f[mstart:mstart + j + 1])
                    j += 1
                except ValueError:
                    mx = float(f[mstart:mstart + j])
                    found_mass = True
            if self.nobs:
                nobs = interpola(mx, sources[:, 0], sources[:, 1])
                if nobs < 0:
                    nobs = 0.
                print 'mx: ', mx, 'Nobs: ', nobs
            else:
                nobs = 0.
            if self.method == "Poisson":
                lim_vals = Poisson(nobs, self.nbkg, self.CL).poisson_integ_up()
                lim_val = lim_vals
            elif self.method == "FeldmanCousins":
                lim_vals = Poisson(self.nobs, self.nbkg, self.CL).FeldmanCousins()
                if lim_vals[0] == 0.:
                    lim_val = lim_vals[1]
                else:
                    print 'Feldman-Cousins returns two sided band... Exiting...'
                    lim_val = 0.
                    exit()
            else:
                print 'Invalid method call.'
                raise ValueError
            cross_vs_n = np.loadtxt(f)
            cs_list = np.logspace(np.log10(cross_vs_n[0, 0]), np.log10(cross_vs_n[-1, 0]), 200)
            cross_n_interp = interp1d(np.log10(cross_vs_n[:, 0]), np.log10(cross_vs_n[:, 1]), kind='cubic')
            fd_min = np.abs(10. ** cross_n_interp(np.log10(cs_list)) - lim_val)
            print 'Cross Section Limit: ', cs_list[np.argmin(fd_min)]

            limarr[i] = [mx, cs_list[np.argmin(fd_min)]]
        limarr = limarr[np.argsort(limarr[:, 0])]
        print 'Limit: '
        print limarr
        np.savetxt(self.folder + file_name, limarr)
        return


class Poisson(object):
    """
    Contains poisson statistics information. Can choosen from Feldman-Cousins or standard Poisson analysis
    """
    def __init__(self, nobs, nbkg=0., CL=0.9):
        self.nobs = nobs
        self.nbkg = nbkg
        self.CL = CL

    def pdf(self, mu, b, n):
        pdf = (mu + b)**n * np.exp(-(mu + b)) / gamma(n + 1.)
        return pdf

    def poisson_integ_up(self):
        norm = quad(self.pdf, 0., np.inf, args=(self.nbkg, self.nobs))[0]

        def min_quant(x):
            intag = quad(self.pdf, x, np.inf, args=(self.nbkg, self.nobs))
            return np.abs(intag[0] / norm - (1. - self.CL))
        return fminbound(min_quant, 0., 100.)

    def FeldmanCousins(self, nmax=60, mu_max=20., n_mu=1000):
        mu_tab = np.linspace(0., mu_max, n_mu)
        fc_tab = np.zeros(3 * nmax).reshape((nmax, 3))
        mu_band = np.zeros(3 * n_mu).reshape((n_mu, 3))

        for i, muv in enumerate(mu_tab):
            for x in range(nmax):
                hatval = np.max([0., x - self.nbkg])
                fc_tab[x] = [x, self.pdf(muv, self.nbkg, x), self.pdf(muv, self.nbkg, x) /
                             self.pdf(hatval, self.nbkg, x)]

            fc_tab = fc_tab[np.argsort(fc_tab[:, 2])[:: -1]]
            sumtot = 0.
            j = 0

            while sumtot < self.CL:
                sumtot += fc_tab[j, 1]
                j += 1

            mu_band[i] = [muv, np.min(fc_tab[0:j, 0]), np.max(fc_tab[0:j, 0])]

        r_found = False
        m_regions = False

        for i in range(mu_band[:, 0].size):
            if mu_band[i, 1] <= self.nobs <= mu_band[i, 2]:
                try:
                    cl_tab = np.append(cl_tab, mu_band[i, 0])
                except NameError:
                    cl_tab = np.array([mu_band[i, 0]])
                if not r_found:
                    r_found = True
                elif r_found and not (mu_band[i-1, 0] in cl_tab):
                    print 'Potentially have multiple disconnected regions'
                    m_regions = True
        if m_regions:
            print 'Full Tab. Look for multiple regions.'
            print cl_tab

        print 'Minimum value of mu: ', np.min(cl_tab)
        print 'Maximum value of mu: ', np.max(cl_tab)

        return [np.min(cl_tab), np.max(cl_tab)]