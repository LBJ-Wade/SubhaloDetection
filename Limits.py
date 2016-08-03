# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

@author: SamWitte
"""
import numpy as np
import os
from math import factorial
from scipy.integrate import quad

class Poisson(object):

    def __init__(self, nobs, nbkg, CL):
        self.nobs = nobs
        self.nbkg = nbkg
        self.CL = CL

    def pdf(self, mu, b, n):
        pdf = (mu + b)**n * np.exp(-(mu + b)) / factorial(n)
        return pdf

    def poisson_integ_up(self):
        norm = quad(self.pdf, 0., np.inf, args=(self.nbkg, self.nobs))[0]
        mutab = np.linspace(0., 20., 1000)
        diff_tab = np.zeros(mutab.size)
        for i,muv in enumerate(mutab):
            intag = quad(self.pdf, muv, np.inf, args=(self.nbkg, self.nobs))
            diff_tab[i] = np.abs(intag[0] / norm - (1. - self.CL))
        return mutab[np.argmin(diff_tab)]

    def FeldmanCousins(self, nmax=60, mu_max=20., n_mu=1000):
        mu_tab = np.linspace(0., mu_max, n_mu)
        fc_tab = np.zeros(3 * nmax).reshape((nmax, 3))
        mu_band = np.zeros(3 * n_mu).reshape((n_mu, 3))

        for i,muv in enumerate(mu_tab):
            for x in range(nmax):
                hatval = np.max([0., x - self.nbkg])
                fc_tab[x] = [x, self.pdf(muv, self.nbkg, x), self.pdf(muv, self.nbkg, x) /
                             self.pdf(hatval, self.nbkg, x)]

            fc_tab = fc_tab[np.argsort(fc_tab[:,2])[:: -1]]
            sumtot = 0.
            j = 0

            while sumtot < self.CL:
                sumtot += fc_tab[j, 1]
                j += 1

            mu_band[i] = [muv, np.min(fc_tab[0:j, 0]), np.max(fc_tab[0:j, 0])]

        r_found = False
        m_regions = False

        for i in range(mu_band[:,0].size):
            if mu_band[i,1] <= self.nobs <= mu_band[i,2]:
                try:
                    cl_tab = np.append(cl_tab, mu_band[i, 0])
                except NameError:
                    cl_tab = np.array([mu_band[i, 0]])
                if not r_found:
                    r_found = True
                elif r_found and not (mu_band[i-1,0] in cl_tab):
                    print 'Potentially have multiple disconnected regions'
                    m_regions = True
        if m_regions:
            print 'Full Tab. Look for multiple regions.'
            print cl_tab

        print 'Minimum value of mu: ',np.min(cl_tab)
        print 'Maximum value of mu: ',np.max(cl_tab)

        return [np.min(cl_tab), np.max(cl_tab)]