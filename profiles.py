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
from scipy.optimize import fminbound
from scipy.interpolate import interp1d


class Subhalo(object):
    """
    Superclass of specific subhalo profiles that calculates the J-factor,
    the spatial extension, and the necessary ingredients to obtain the
    flux threshold for spatially extended sources
    """
    def J(self, dist, theta):
        """
        Calculates J factor.
        :param dist: Distance to subhalo in kpc
        :param theta: Upper bound on anuglar integration (in degrees)
        :return: returns log10 of J factor
        """
        max_theta = radtodeg * np.arctan(self.max_radius / dist)
        if theta > max_theta:
            theta = max_theta
        theta = theta * np.pi / 180.

        def eps(th, d):
            return (self.los_max(th, d) - self.los_min(th, d)) / 10. ** 4.

        if theta > 0.01 * np.pi / 180:
            jfact = integrate.dblquad(lambda x, t:
                                      2. * np.pi * kpctocm * np.sin(t) *
                                      self.density(np.sqrt(dist ** 2. + x ** 2. -
                                                           2.0 * dist * x * np.cos(t))
                                                   ) ** 2.0,
                                      0., theta, lambda th: self.los_min(th, dist),
                                      lambda th: self.los_max(th, dist), epsabs=10 ** -4,
                                      epsrel=10 ** -4)

            return np.log10(jfact[0])
        else:
            return self.J_pointlike(dist)

    def J_pointlike(self, dist):
        """
        Calculates J factor, assuming its point-like.
        :param dist: Distance to subhalo in kpc
        :return: returns log10 of J factor in GeV^2 /
        """
        jfact = self.int_over_rho_sqr(self.max_radius) / dist**2.
        return np.log10(jfact)

    def los_min(self, theta, dist):
        """
        Calculates lower bound on los integral
        :param theta: Angle from subhalo center in degrees
        :param dist: Distance to subhalo in kpc
        :return: min bound of los integration
        """
        return dist * np.cos(theta) - np.sqrt(dist ** 2. * (np.cos(theta) ** 2. - 1.)
                                              + self.max_radius ** 2.)

    def los_max(self, theta, dist):
        """
        Calculates upper bound on los integral
        :param theta: Angle from subhalo center in degrees
        :param dist: Distance to subhalo in kpc
        :return: max bound of los integration
        """
        return dist * np.cos(theta) + np.sqrt(dist ** 2. * (np.cos(theta) ** 2. - 1.)
                                              + self.max_radius ** 2.)

    def Mass_in_R(self, r):
        """
        Calculates total mass (in SM) contained within some radius r (in kpc)
        :param r: upper limit of radial integration
        :return: Mass enclosed
        """
        return self.int_over_density(r)

    def Mass_diff_005(self, rmax):
        """
        Function used for calculating the truncated radius. Calculates the difference between
        the mass contained in rmax and the 0.5% of the total subhalo mass
        :param rmax: upper bound on radial integration in kpc
        :return: Mass difference
        """
        rmax = 10**rmax
        mass_enc = self.int_over_density(rmax)
        return np.abs(mass_enc - 0.005 * self.halo_mass)

    def Truncated_radius(self):
        """
        Calculates the radius containing 0.5% of the total subhalo mass
        :return: Value of truncated radius in kpc
        """
        r_trunc = fminbound(self.Mass_diff_005, -10., np.log10(self.scale_radius))
        return 10**float(r_trunc)

    def AngRad68(self, theta, dist):
        """
        Function used for calculating the quoted angular extension -- used to
        obtain flux thresholds for spatially extended sources
        :param theta: Max angle of integration in degrees
        :param dist: Subhalo distance in kpc
        :return: In principle we would like to find the angle theta at which the ratio of
        J(d,t)/J_pointlike(d) = 0.68. To facilitate this (and use more manageable numbers)
        we take the log10 of each side and minimize the abs value.
        """
        return np.abs(10. ** self.J(dist, theta) / 10. ** self.J_pointlike(dist) - 0.68)

    def Spatial_Extension(self, dist):
        """
        Function that minimizes AngRad68
        :param dist: Distance of subhalo in kpc
        :return: returns quoted spatial extension to be used for calculation of
        flux threshold of spatially extended sources
        """
        extension = fminbound(self.AngRad68, 10**-2., radtodeg *
                              np.arctan(self.max_radius / dist), args=[dist],
                              xtol=10**-3.)
        return extension

    def Full_Extension(self, dist):
        """
        Calculates the maximum extension (based on max radius)
        of a subhalo at distance = dist
        :param dist: Distance in kpc
        :return: returns extension in degrees
        """
        return radtodeg * np.arctan(self.max_radius / dist)

    def find_tidal_radius(self, r):
        return np.abs(self.halo_mass - self.int_over_density(10. ** r))


class Einasto(Subhalo):
    """
    Class sets all of information for subhalos fitted to Einasto profile.


    halo_mass: Subhalo mass (if truncate = True, this mass is pre-tidal stripping mass)

    alpha: Exponent parameter in density profile

    concentration_param: Concentration parameter. If set to None, it uses c(M).

    z: Redshift (currently unused but included because one of concentration
    parameter sources has possible z dependence)

    truncate: If True, analysis follows Hooper et al. If False, subhalo is
    assumed to be fit by Einasto profile after tidal stripping

    arxiv_num: Arxiv number of paper parameterizing the concentration
    parameter -- currently only [10070438, 13131729] implemented

    M200: If False, virial radius is scaled version of Milky Ways virial radius.
    If True, virial radius is taken to be the radius at which the average density
    is equal to 200 times the critical density.
    """
    def __init__(self, halo_mass, alpha, concentration_param=None,
                 z=0., truncate=False, arxiv_num=10070438, M200=False):

        self.pname = 'Einasto_alpha_' + str(alpha) + '_C_params_' + str(arxiv_num) + \
            '_Truncate_' + str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha
        self.halo_name = 'Einasto'
        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)
        self.c = concentration_param

        self.virial_radius = Virial_radius(self.halo_mass, m200=M200)

        if arxiv_num == 160106781:
            rmax, vmax = rmax_vmax(self.halo_mass)
            self.scale_radius = (rmax / 1.21) / 2.163
            self.scale_density = ((vmax ** 2. * rmax * np.exp(-2. / self.alpha) * 8. ** (1. / self.alpha)) /
                                  (4. * np.pi * newton_G * self.scale_radius ** 2. *
                                   self.alpha ** (3. / self.alpha) * special.gamma(3. / self.alpha) *
                                   (1. - special.gammaincc(3. / self.alpha, 2. *
                                                           (rmax / self.scale_radius) ** self.alpha / self.alpha))) *
                                  SolarMtoGeV * cmtokpc ** 3.)

        else:

            self.scale_radius = self.virial_radius / self.c
            self.scale_density = ((self.halo_mass * self.alpha * np.exp(-2. / self.alpha) *
                                   (2. / self.alpha) ** (3. / self.alpha)) /
                                  (4. * np.pi * self.scale_radius ** 3. *
                                   special.gamma(3. / self.alpha) *
                                   (1. - special.gammaincc(3. / self.alpha, 2. *
                                                           self.c ** self.alpha / self.alpha))) *
                                  SolarMtoGeV * cmtokpc ** 3.)

        if not truncate:
            if arxiv_num == 160106781:
                self.max_radius = np.power(10, fminbound(self.find_tidal_radius, -4., 1.3))
            else:
                self.max_radius = self.virial_radius
        else:
            self.max_radius = self.Truncated_radius()

    def density(self, r):
        return self.scale_density * np.exp(-2. / self.alpha * (((r / self.scale_radius) **
                                                                self.alpha) - 1.))

    def int_over_density(self, r):
        if r > 0:
            return ((4. * np.pi * np.exp(2. / self.alpha) * self.scale_density *
                    (self.alpha / 2.) ** (3. / self.alpha) * self.scale_radius ** 3. *
                    special.gamma(3. / self.alpha) *
                    (1. - special.gammaincc(3. / self.alpha, 2. / self.alpha *
                                            (r / self.scale_radius) ** self.alpha))) *
                    kpctocm ** 3. * GeVtoSolarM / self.alpha)

    def int_over_rho_sqr(self, r):
        if r > 0:
            return ((4. * np.pi * np.exp(4. / self.alpha) * self.scale_density ** 2. *
                    self.alpha ** (3. / self.alpha - 1.) * self.scale_radius ** 3. *
                    special.gamma(3. / self.alpha) *
                    (1. - special.gammaincc(3. / self.alpha, 4. / self.alpha *
                                            (r / self.scale_radius) ** self.alpha))) *
                    kpctocm / 4. ** (3. / self.alpha))


class NFW(Subhalo):
    """
    Class sets all of information for subhalos fitted to NFW profile.


    halo_mass: Subhalo mass (if truncate = True, this mass is pre-tidal stripping mass)

    alpha: Does not do anything -- I'm keeping it because I have some ideas...

    concentration_param: Concentration parameter. If set to None, it uses c(M).

    z: Redshift (currently unused but included becuase one of concentration
    paramater sources has possible z dependence)

    truncate: If True, analysis follows Hooper et al. If False, subhalo is
    assumed to be fit by Einasto profile after tidal stripping

    arxiv_num: Arxiv number of paper parameterizing the concentration
    parameter -- currently only [10070438, 13131729] implemented

    M200: If False, virial radius is scaled version of Milky Ways virial radius.
    If True, virial radius is taken to be the radius at which the average density
    is equal to 200 times the critical density.
    """
    def __init__(self, halo_mass, alpha, concentration_param=None,
                 z=0., truncate=False, arxiv_num=10070438, M200=False):

        self.pname = 'NFW_alpha_' + '_C_params_' + str(arxiv_num) + \
                     '_Truncate_' + str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha
        self.halo_name = 'NFW'
        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)
        self.c = concentration_param
        self.virial_radius = Virial_radius(self.halo_mass, m200=M200)

        if arxiv_num == 160106781:
            rmax, vmax = rmax_vmax(self.halo_mass)
            self.scale_radius = (rmax / 1.21) / 2.163
            self.scale_density = (2.163 * vmax ** 2. / (4. * np.pi * newton_G * self.scale_radius ** 2. *
                                                       (np.log(3.163) - 2.163 / 3.163)) * SolarMtoGeV *
                                  cmtokpc ** 3.)
        else:
            self.scale_radius = self.virial_radius / self.c
            self.scale_density = ((self.halo_mass * SolarMtoGeV * cmtokpc ** 3.) /
                              (4. * np.pi * self.scale_radius ** 3. *
                               (np.log(1.0 + self.c) -
                                1.0 / (1.0 + 1.0 / self.c))))

        if not truncate:
            if arxiv_num == 160106781:
                self.max_radius = np.power(10, fminbound(self.find_tidal_radius, -4., 1.3))
            else:
                self.max_radius = self.virial_radius
        else:
            self.max_radius = self.Truncated_radius()

    def density(self, r):
        return self.scale_density / ((r / self.scale_radius) * (1. + r / self.scale_radius) ** 2.)

    def int_over_density(self, r):
        if r > 0:
            return (self.scale_density * 4. * np.pi * self.scale_radius ** 3. *
                    (np.log((r + self.scale_radius) / self.scale_radius) - r / (r + self.scale_radius)) *
                    kpctocm ** 3. * GeVtoSolarM)

    def int_over_rho_sqr(self, r):
        if r > 0:
            return (4. * np.pi * self.scale_density ** 2. * self.scale_radius ** 3. / 3. *
                    (1. - 1. / (1. + r / self.scale_radius) ** 3.) * kpctocm)

