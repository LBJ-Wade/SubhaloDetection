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
        if (self.los_max(0., dist) - self.los_max(theta, dist)) > 10 ** -5.:
            jfact = integrate.dblquad(lambda x, t:
                                      2. * np.pi * kpctocm * np.sin(t) *
                                      self.density(np.sqrt(dist ** 2. + x ** 2. -
                                                           2.0 * dist * x * np.cos(t))
                                                   ) ** 2.0,
                                      0., theta, lambda x: self.los_min(x, dist),
                                      lambda x: self.los_max(x, dist), epsabs=10 ** -5,
                                      epsrel=10 ** -5)

            return np.log10(jfact[0])
        else:
            return self.J_pointlike(dist)

    def J_pointlike(self, dist):
        """
        Calculates J factor, assuming its pointlike.
        :param dist: Distance to subhalo in kpc
        :return: returns log10 of J factor
        """
        jfact = integrate.quad(lambda x: 4. * np.pi * kpctocm / dist ** 2. *
                               self.density(x) ** 2. * x ** 2.,
                               0., self.max_radius, epsabs=10 ** -5, epsrel=10 ** -5)

        return np.log10(jfact[0])

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
        mass_enc = integrate.quad(lambda x: 4. * np.pi * x ** 2. * self.density(x) *
                                  GeVtoSolarM * kpctocm ** 3., 0., r)
        return mass_enc[0]

    def Mass_diff_005(self, rmax):
        """
        Function used for calculating the truncated radius. Calculates the difference between
        the mass contained in rmax and the 0.5% of the total subhalo mass
        :param rmax: upper bound on radial integration in kpc
        :return: Mass difference
        """
        rmax = 10**rmax
        mass_enc = integrate.quad(lambda x: x ** 2. * self.density(x), 0., rmax)
        return np.abs(4. * np.pi * GeVtoSolarM * kpctocm ** 3. *
                      mass_enc[0] - 0.005 * self.halo_mass)

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
        J(d,t)/J_pointlike(d) = 0.68. To facilitiate this (and use more manageable numbers)
        we take the log10 of each side and minimize the abs value.
        """
        return np.abs(self.J(dist, theta) - self.J_pointlike(dist) - np.log10(0.68))

    def Spatial_Extension(self, dist):
        """
        Function that minimizes AngRad68
        :param dist: Distance of subhalo in kpc
        :return: returns quoted spatial extension to be used for calculation of
        flux threhosld of spatially extended sources
        """
        extension = fminbound(self.AngRad68, 10**-7., radtodeg *
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


class Einasto(Subhalo):
    """
    Class sets all of information for subhalos fitted to Einasto profile.


    halo_mass: Subhalo mass (if truncate = True, this mass is pre-tidal stripping mass)

    alpha: Exponent parameter in density profile

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

        self.pname = 'Einasto_alpha_' + str(alpha) + '_C_params_' + str(arxiv_num) + \
            '_Truncate_' + str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha

        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)
        self.c = concentration_param
        if M200:
            self.virial_radius = (3. * self.halo_mass / (4. * np.pi * rho_critical
                                                         * delta_200))**(1. / 3.)
        else:
            self.virial_radius = Virial_radius(self.halo_mass)
        self.scale_radius = self.virial_radius / self.c
        self.scale_density = ((self.halo_mass * self.alpha * np.exp(-2. / self.alpha) *
                               (2. / self.alpha) ** (3. / self.alpha)) /
                              (4. * np.pi * self.scale_radius ** 3. *
                               special.gamma(3. / self.alpha) *
                               (1. - special.gammaincc(3. / self.alpha, 2. *
                                self.c ** self.alpha / self.alpha))) *
                              SolarMtoGeV * cmtokpc ** 3.)

        if not truncate:
            self.max_radius = self.virial_radius
        else:
            self.max_radius = self.Truncated_radius()

    def density(self, r):
        return self.scale_density * np.exp(-2. / self.alpha * (((r / self.scale_radius) **
                                                                self.alpha) - 1.))


class NFW(Subhalo):
    """
    Class sets all of information for subhalos fitted to NFW profile.


    halo_mass: Subhalo mass (if truncate = True, this mass is pre-tidal stripping mass)

    alpha: Does not do antying -- I'm keeping it becuase I have some ideas...

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

        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)

        self.c = concentration_param
        if M200:
            self.virial_radius = (3. * self.halo_mass / (4. * np.pi * rho_critical *
                                                         delta_200))**(1. / 3.)
        else:
            self.virial_radius = Virial_radius(self.halo_mass)
        self.scale_radius = self.virial_radius / self.c
        self.scale_density = ((self.halo_mass * SolarMtoGeV * cmtokpc ** 3.) /
                              (4. * np.pi * self.scale_radius ** 3. *
                               (np.log(1.0 + self.c) -
                                1.0 / (1.0 + 1.0 / self.c))))

        if not truncate:
            self.max_radius = self.virial_radius
        else:
            self.max_radius = self.Truncated_radius()

    def density(self, r):
        if r > 0:
            return self.scale_density / ((r / self.scale_radius) * (1. + r / self.scale_radius) ** 2.)
        else:
            return 0.
