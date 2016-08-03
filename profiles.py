# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

@author: SamWitte
"""
import numpy as np
import os
from subhalo import *
from helper import *
from Limits import *
import scipy.integrate as integrate
import scipy.special as special
from scipy.optimize import minimize,brute



class Subhalo(object):
    def J(self, dist, theta):
        """Theta in degrees and distance in kpc"""

        if theta > radtodeg * np.arctan(self.max_radius / dist):
            theta = np.arctan(self.max_radius / dist)
        else:
            theta = theta * np.pi / 180.

        if (self.los_max(0., dist) - self.los_max(theta, dist)) > 10 ** -5:

            jfact = integrate.dblquad(lambda x, t: 2. * np.pi * kpctocm * np.sin(t) *
                                                   self.density(np.sqrt(dist ** 2. + x ** 2. - 2.0 *
                                                                        dist * x * np.cos(t))) ** 2.0, 0., theta,
                                      lambda x: self.los_min(x, dist),
                                      lambda x: self.los_max(x, dist), epsabs=10 ** -5,
                                      epsrel=10 ** -5)

            return np.log10(jfact[0])
        else:
            return self.J_pointlike(dist)

    def J_pointlike(self, dist):

        jfact = integrate.quad(lambda x: 4. * np.pi * kpctocm / dist ** 2. * self.density(x) ** 2. * x ** 2.,
                               0., self.max_radius, epsabs=10 ** -5, epsrel=10 ** -5)

        return np.log10(jfact[0])

    def los_min(self, theta, dist):
        return dist * np.cos(theta) - np.sqrt(dist ** 2. * (np.cos(theta) ** 2. - 1.) + self.max_radius ** 2.)

    def los_max(self, theta, dist):
        return dist * np.cos(theta) + np.sqrt(dist ** 2. * (np.cos(theta) ** 2. - 1.) + self.max_radius ** 2.)

    def Mass_in_R(self, r):
        mass_enc = integrate.quad(lambda x: 4. * np.pi * x ** 2. * self.density(x) *
                                            GeVtoSolarM * (kpctocm) ** 3., 0., r)
        return mass_enc[0]

    def Mass_diff_005(self, rmax):
        mass_enc = integrate.quad(lambda x: x ** 2. * self.density(x), 0., rmax)
        return np.abs(4. * np.pi * GeVtoSolarM * (kpctocm) ** 3. * mass_enc[0] - 0.005 * self.halo_mass)

    def Truncated_radius(self):
        bnds = [(10 ** -5, self.scale_radius)]
        #       rtrunc = minimize(self.Mass_diff_005, [0.5 * self.scale_radius], bounds=bnds)
        rtrunc = brute(self.Mass_diff_005, bnds)
        return rtrunc

    def AngRad68(self, theta, dist):
        return np.abs(self.J(dist, theta) - self.J_pointlike(dist) - np.log10(0.68))

    def Spatial_Extension(self, dist):
        bnds = [(10 ** -4., radtodeg * np.arctan(self.max_radius / dist))]
        extension = minimize(self.AngRad68, np.array([0.9 * radtodeg * np.arctan(self.max_radius / dist)]),
                             args=(dist), method='SLSQP', bounds=bnds, tol=10 ** -2)
        return extension.x

    def Full_Extension(self, dist):
        return radtodeg * np.arctan(self.max_radius / dist)

class Einasto(Subhalo):

    def __init__(self, halo_mass, alpha, concentration_param=None,
                 z=0., truncate=False, arxiv_num=10070438):

        self.pname = 'Einasto_alpha_'+ str(alpha) + '_C_params_' + str(arxiv_num) + \
                     '_Truncate_' + str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha

        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)

        self.c = concentration_param
        if arxiv_num == 10070438:
            M200 = False
        elif arxiv_num == 13131729:
            M200 = True
        # M200 = False
        self.scale_radius = Virial_radius(self.halo_mass, M200=M200) / self.c
        self.scale_density = ((self.halo_mass * self.alpha * np.exp(-2. / self.alpha) *
                               (2. / self.alpha) ** (3. / self.alpha)) / (4. * np.pi *
                                                                          self.scale_radius ** 3. * special.gamma(
            3. / self.alpha) *
                                                                          (1. - special.gammaincc(3. / self.alpha,
                                                                                                  2. * self.c ** self.alpha /
                                                                                                  self.alpha))) * SolarMtoGeV * (
                              cmtokpc) ** 3.)

        if not truncate:
            self.max_radius = self.scale_radius
        else:
            self.max_radius = self.Truncated_radius()

    def density(self, r):
        return self.scale_density * np.exp(-2. / self.alpha * (((r / self.scale_radius) ** self.alpha) - 1.))


class NFW(Subhalo):
    def __init__(self, halo_mass, alpha, concentration_param=None,
                 z=0., truncate=False, arxiv_num=10070438):
        """Note: alpha doesn't do anything, I'm keeping it here to be
           consistent with Einasto profile
        """

        self.pname = 'NFW_alpha_' + '_C_params_' + str(arxiv_num) + \
                     '_Truncate_' + str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha

        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)

        self.c = concentration_param

        if arxiv_num == 13131729:
            M200 = False
        # M200 = True
        else:
            M200 = False

        self.scale_radius = Virial_radius(self.halo_mass, M200=M200) / self.c
        self.vir_rad = Virial_radius(self.halo_mass, M200=M200)
        self.scale_density = ((self.halo_mass * SolarMtoGeV * (cmtokpc) ** 3.) /
                              (4. * np.pi * self.scale_radius ** 3. *
                               (np.log(self.scale_radius + self.vir_rad) -
                                np.log(self.scale_radius) - self.vir_rad /
                                (self.scale_radius + self.vir_rad))))

        if not truncate:
            self.max_radius = self.scale_radius
        else:
            self.max_radius = self.Truncated_radius()

    def density(self, r):
        return self.scale_density / ((r / self.scale_radius) * (1. + r / self.scale_radius) ** 2.)
