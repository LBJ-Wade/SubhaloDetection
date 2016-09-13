# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from helper import *
from Limits import *
from profiles import *
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import RectBivariateSpline,interp1d,interp2d,\
    LinearNDInterpolator,bisplrep, bisplev
from scipy.optimize import fminbound
import os
import pickle
import glob
from Plotting_Functions import *
from Joint_Sim_Comparison import *
import time



try:
    MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'


class Model(object):
    """
    Given a particular DM model and halo, this can calculate various properties such as the
    maximum distance from Earth such a candidate can be to still be detectable by FermiLat
    (or the flux, spatially extended threshold, etc)
    """
    def __init__(self, mx, cross_sec, annih_prod, halo_mass, alpha=.16,
                 concentration_param=None, z=0., truncate=False,
                 arxiv_num=13131729, profile=0, pointlike=False,
                 m200=False, gam=0.85, stiff_rb=False, rb=None):

        self.mx = mx
        self.c_sec = cross_sec
        self.annih_prod = annih_prod
        self.arxiv_num = arxiv_num
        self.truncate = truncate
        self.alpha = alpha
        self.profile = profile
        self.plike = str2bool(pointlike)
        self.gamma = self.Determine_Gamma()
        self.m200 = m200
        self.stiff_rb = stiff_rb
        self.rb = rb
        self.gam = gam
        self.halo_mass = halo_mass
        self.c = concentration_param

        if profile == 0:
            self.subhalo = Einasto(halo_mass, alpha, 
                                   concentration_param=concentration_param, 
                                   truncate=truncate, arxiv_num=arxiv_num,
                                   M200=self.m200)
        elif profile == 1:
            self.subhalo = NFW(halo_mass, alpha, 
                               concentration_param=concentration_param, 
                               truncate=truncate, arxiv_num=arxiv_num,
                               M200=self.m200)
        elif profile == 2:
            self.subhalo = HW_Fit(halo_mass, gam=gam, gcd=8.5, M200=self.m200,
                                  stiff_rb=self.stiff_rb, rb=self.rb)

        else:
            "Profile does not exist..."  
            exit()                   

    def min_Flux(self, dist):
        """
        Wrapper that collects spatial extension and threshold information and returns
        flux threshold for spatially extended sources
        :param dist: distance kpc
        """

        file_name = file_name = 'SpatialExtension_' + Profile_list[self.profile] + '.dat'
        dir = MAIN_PATH + '/SubhaloDetection/Data/'
        try:
            se_file = np.loadtxt(dir + file_name)
            m_comp = float('{:.4e}'.format(self.halo_mass))
            if np.sum(se_file[:, 0] == m_comp) > 0:
                se_file = se_file[se_file[:, 0] == m_comp]
                # Only Look at Masses of interest
                if self.profile == 0:
                    c_comp = float('{:.3e}'.format(self.c))
                    # See if C value has been calculated
                    if np.sum(se_file[:, 1] == c_comp) > 0.:
                        halo_of_int = se_file[se_file[:, 1] == c_comp]
                        if halo_of_int[0, -1] == 2.:
                            upper_lim_hit = True
                        else:
                            upper_lim_hit = False
                        if halo_of_int[-1, -1] == 0.05:
                            lower_lim_hit = True
                        else:
                            lower_lim_hit = False
                        valid_dist = halo_of_int[(halo_of_int[:, -1] > 0.05) & (halo_of_int[:, -1] < 2.0)]
                        if (dist < valid_dist[0, -2]) and upper_lim_hit:
                            extension = 2.
                        elif (dist > valid_dist[-1, -2]) and lower_lim_hit:
                            extension = 0.05
                        else:
                            extension = interpola(dist, valid_dist[:, -2], valid_dist[:, -1])
                    else:
                        raise ValueError
                elif self.profile == 1:
                    if se_file[0, -1] == 2.:
                        upper_lim_hit = True
                    else:
                        upper_lim_hit = False
                    if se_file[-1, -1] == 0.05:
                        lower_lim_hit = True
                    else:
                        lower_lim_hit = False
                    valid_dist = se_file[(se_file[:, -1] > 0.05) & (se_file[:, -1] < 2.0)]

                    if (dist < valid_dist[0, -2]) and upper_lim_hit:
                        extension = 2.
                    elif (dist > valid_dist[-1, -2]) and lower_lim_hit:
                        extension = 0.5
                    else:
                        extension = interpola(dist, valid_dist[:, -2], valid_dist[:, -1])

                else:
                    rb_comp = float('{:.3e}'.format(self.rb))
                    g_comp = float('{:.6f}'.format(self.gam))
                    if np.sum((se_file[:, 1] == rb_comp) & (se_file[:, 2] == g_comp)) > 0:
                        halo_of_int = se_file[(se_file[:, 1] == rb_comp) & (se_file[:, 2] == g_comp)]
                        if halo_of_int[0, -1] == 2.:
                            upper_lim_hit = True
                        else:
                            upper_lim_hit = False
                        if halo_of_int[-1, -1] == 0.05:
                            lower_lim_hit = True
                        else:
                            lower_lim_hit = False
                        valid_dist = halo_of_int[(halo_of_int[:, -1] > 0.5) & (halo_of_int[:, -1] < 2.0)]
                        if (dist < valid_dist[0, -2]) and upper_lim_hit:
                            extension = 2.
                        elif (dist > valid_dist[-1, -2]) and lower_lim_hit:
                            extension = 0.05
                        else:
                            extension = interpola(dist, halo_of_int[:, -2], halo_of_int[:, -1])
                    else:
                        raise ValueError
            else:
                raise ValueError
        except:
            extension = self.subhalo.Spatial_Extension(dist)
        return self.Threshold(self.gamma, extension)

    def Total_Flux(self, dist):
        """
        Returns total flux from subhalo
        :param dist: distance of subhalo in kpc
        """
        pre_factor = self.c_sec / (8. * np.pi * self.mx**2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
            self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1], kind='cubic')
        n_gamma = integrated_rate(self.mx)
        flux = pre_factor * n_gamma * 10**self.subhalo.J_pointlike(dist)
        return flux
        
    def Partial_Flux(self, dist, theta):
        """
        Returns partial flux of threshold based on integrating out to some theta less than theta_max
        :param dist: distance of subhalo in kpc
        :param theta: integration bound
        :return: partial flux
        """
        pre_factor = self.c_sec / (8. * np.pi * self.mx**2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
            self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1], kind='cubic')
        n_gamma = integrated_rate(self.mx)
        flux = pre_factor * n_gamma * 10**self.subhalo.J(dist, theta)
        return flux

    def d_max_point(self, threshold=(7.0 * 10 ** (-10.))):
        """
        Calculates the maximum distance a point-like subhalo can be to still be observable,
        given a particular threshold
        :param threshold: flux threshold in photons / cm^2 / s
        :return: distance in kpc
        """
        pre_factor = self.c_sec / (8. * np.pi * self.mx ** 2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
            self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        n_gamma = interpola(self.mx, integrated_list[:, 0], integrated_list[:, 1])
        jf = 10. ** self.subhalo.J_pointlike(1.)
        return np.sqrt(pre_factor * n_gamma * jf / threshold)

    def D_max_extend(self):
        """
        Calculates the maximum distance a spatially extended subhalo can be to still be observable
        :return: distance in kpc
        """
        max_dist = self.d_max_point()
        dist_tab = np.logspace(-1., np.log10(max_dist), 15)
        flux_diff_tab = np.zeros(dist_tab.size)
        for i, d in enumerate(dist_tab):
            flux_diff_tab[i] = np.abs(self.Total_Flux(d) - self.min_Flux(d))
        #interp = interpola(full_dist_tab, dist_tab, flux_diff_tab)
        cent = np.argmin(flux_diff_tab)
        try:
            full_dist_tab = np.logspace(np.log10(dist_tab[0]), np.log10(dist_tab[cent + 2]), 60)
            interp = interpola(full_dist_tab, dist_tab[:cent + 2], flux_diff_tab[:cent + 2])
        except:
            full_dist_tab = np.logspace(np.log10(dist_tab[0]), np.log10(dist_tab[-1]), 60)
            interp = interp1d(dist_tab, flux_diff_tab, kind='linear', fill_value=10**5,
                              bounds_error=False)(full_dist_tab)
        d_max = full_dist_tab[np.argmin(interp)]
        #def flux_diff_lten(x):
        #    return np.abs(self.Total_Flux(10. ** x) - self.min_Flux(10. ** x))
        #d_max = fminbound(flux_diff_lten, -4., 2., xtol=10**-4.)
        #return 10 ** d_max
        return d_max

    def Determine_Gamma(self):
        """
        Caclualtes the spectral index to be used to determine the spatial extended threshold
        :return: [1.5, 2.0, 2.5, 3.0] whichever spectral index most closely produces same average
        photon energy
        """

        gamma_tab = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/GammaIndex_given_mx_for_annih_prod_' +
                               self.annih_prod + '.dat')
        gamma = interpola(self.mx, gamma_tab[:, 0], gamma_tab[:, 1])

        def myround(x, prec=2, base=.5):
            return round(base * round(float(x) / base), prec)
        gamma_approx = myround(gamma)
        return gamma_approx

    def Threshold(self, gamma, extension):
        """
        Calcualtes the flux theshold in (photons / cm^2 / s) of a DM candidate

        :param gamma: Either 1.5, 2.0, 2.5, 3.0 -- found by determining the spectral index
        which most closely reproduces the average photon energy
        :param extension: (Radial) Spatial extension of subhalo in degrees
        :return: Flux threshold
        """
        file = MAIN_PATH + "/ExtendedThresholds/DetectionThresholdGamma" + \
            str(int(gamma * 10)) + ".dat"

        thresh_list = np.loadtxt(file)
        if extension > 2.0:
            return interpola(2.0, thresh_list[:, 0], thresh_list[:, 1])
        else:
            return interpola(extension, thresh_list[:, 0], thresh_list[:, 1])


class Observable(object):
    """
    The main purpose of this class is to calculate the number of observable point-like or
    spatially extended sources for a particular DM candidate (ie DM mass, cross section,
    and annihilation products have to be specified)
    """

    def __init__(self, mx, cross_sec, annih_prod, m_low=np.log10(3.24 * 10**4.),
                 m_high=np.log10(1.0 * 10 ** 7.), c_low=np.log10(2.),
                 c_high=2.1, alpha=0.16, profile=0, truncate=False, 
                 arxiv_num=10070438, point_like=True, m200=False,
                 gam=0.85, stiff_rb=False, mltag=None):

        point_like = str2bool(point_like)
        stiff_rb = str2bool(stiff_rb)
        self.truncate = truncate
        self.mx = mx
        self.cross_sec = cross_sec
        self.annih_prod = annih_prod
        self.c_low = c_low
        self.c_high = c_high
        self.alpha = alpha
        self.profile = profile
        self.profile_name = Profile_list[self.profile]
        self.m200 = m200
        self.gam = gam
        self.stiff_rb =stiff_rb
        self.arxiv_num = arxiv_num

        if self.truncate:
            self.m_low = np.log10(10. ** m_low / 0.005)
            self.m_high = np.log10(10. ** m_high / 0.005)
            self.tr_tag = '_Truncate_'
        else:
            self.m_low = m_low
            self.m_high = m_high
            self.tr_tag = ''
        self.point_like = point_like
        if self.point_like:
            ptag = '_Pointlike'
        else:
            ptag = '_Extended'

        if self.profile == 2:
            self.extra_tag = '_gamma_{:.2f}_Conser_'.format(self.gam) + '_stiff_rb_' + str(self.stiff_rb)
        else:
            self.extra_tag = '_arxiv_' + str(self.arxiv_num)
        if mltag is None:
            mltag = m_low
        self.arxiv_num = arxiv_num
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"
        info_str = "Observable_Profile_" + self.profile_name + self.tr_tag +\
            ptag + "_mx_" + str(mx) + "_annih_prod_" +\
            annih_prod + self.extra_tag + '_Mlow_{:.3e}'.format(mltag) + "/"
        
        self.folder += info_str
        ensure_dir(self.folder)
        # TODO: Decide if dictionary is useless or needs an upgrade...
        default_dict = {'profile': 'Einasto', 'truncate': False, 'mx': 100., 'alpha': 0.16,
                        'annih_prod': 'BB', 'arxiv_num': 10070438, 'c_low': np.log10(20.),
                        'c_high': 2.1, 'c_num': 15, 'm_low': np.log10(3.24 * 10 ** 4.),
                        'm_high': np.log10(1.0 * 10 ** 7), 'm_num': 30, 'gamma': 0.85,
                        'stiff_rb': False}

        self.param_list = default_dict
        self.param_list['profile'] = self.profile_name
        self.param_list['truncate'] = self.truncate
        self.param_list['mx'] = self.mx
        self.param_list['annih_prod'] = self.annih_prod
        self.param_list['arxiv_num'] = self.arxiv_num
        self.param_list['c_low'] = self.c_low
        self.param_list['c_high'] = self.c_high
        self.param_list['m_low'] = self.m_low 
        self.param_list['m_high'] = self.m_high
        self.param_list['gamma'] = self.gam
        self.param_list['stiff_rb'] = self.stiff_rb

    def Table_Dmax(self, m_num=20, c_num=15, threshold=7.*10.**-10., plike=True):
        """
        Tables the maximimum distance a source can be detected for a specified
        threshold given a particular DM candidate and subhalo specificiations
        """
        self.param_list['m_num'] = m_num
        self.param_list['c_num'] = c_num

        if plike:
            file_name = 'Dmax_POINTLIKE_' + str(Profile_list[self.profile]) + self.tr_tag +\
                        '_mx_' + str(self.mx) + '_cross_sec_{:.3e}'.format(self.cross_sec) +\
                        '_annih_prod_' + self.annih_prod + self.extra_tag + '.dat'
            mass_list = np.logspace(self.m_low, self.m_high, (self.m_high - self.m_low) * 6)
        else:
            file_name = 'Dmax_Extended_' + str(Profile_list[self.profile]) + self.tr_tag + \
                        '_mx_' + str(self.mx) + '_cross_sec_{:.3e}'.format(self.cross_sec) + \
                        '_annih_prod_' + self.annih_prod + self.extra_tag + '.dat'
            mass_list = np.logspace(self.m_low, self.m_high, 1)
        print 'Cross Section: ', self.cross_sec, '\n'
        for m in mass_list:
            print 'Subhalo mass: ', m
            if self.profile == 0:
                c_list = np.logspace(self.c_low, self.c_high, c_num)
                for c in c_list:
                    print '    Concentration parameter: ', c
                    if self.truncate:
                        mm = 0.005 * m
                    else:
                        mm = m
                    try:
                        look_array = np.loadtxt(self.folder + file_name)
                        mlook = float('{:.3e}'.format(mm))
                        clook = float('{:.3e}'.format(c))
                        if np.sum((mlook == look_array[:, 0]) & (clook == look_array[:, 1])) >= 1:
                            exists = True
                        else:
                            exists = False
                    except:
                        exists = False
                    if not exists:
                        dm_model = Model(self.mx, self.cross_sec, self.annih_prod,
                                         m, self.alpha, concentration_param=c,
                                         truncate=self.truncate,
                                         arxiv_num=self.arxiv_num,
                                         profile=self.profile, pointlike=self.point_like,
                                         m200=self.m200)

                        if self.point_like:
                            dmax = dm_model.d_max_point(threshold=threshold)
                        else:
                            dmax = dm_model.D_max_extend()
                        tab = np.array([mm, c, dmax]).transpose()
                        if os.path.isfile(self.folder+file_name):
                            load_info = np.loadtxt(self.folder + file_name)
                            add_to_table = np.vstack((load_info, tab))
                            np.savetxt(self.folder + file_name, add_to_table, fmt='%.3e')
                        else:
                            np.savetxt(self.folder + file_name, tab, fmt='%.3e')
            elif self.profile == 1:
                try:
                    look_array = np.loadtxt(self.folder + file_name)
                    mlook = float('{:.3e}'.format(m))
                    if mlook in look_array[:, 0]:
                        exists = True
                    else:
                        exists = False
                except:
                    exists = False
                if not exists:
                    dm_model = Model(self.mx, self.cross_sec, self.annih_prod,
                                     m, self.alpha,
                                     truncate=self.truncate,
                                     arxiv_num=self.arxiv_num,
                                     profile=self.profile, pointlike=self.point_like,
                                     m200=self.m200)
                    if self.point_like:
                        dmax = dm_model.d_max_point(threshold=threshold)
                    else:
                        dmax = dm_model.D_max_extend()
                    tab = np.array([m, dmax]).transpose()
                    if os.path.isfile(self.folder + file_name):
                        load_info = np.loadtxt(self.folder + file_name)
                        add_to_table = np.vstack((load_info, tab))
                        np.savetxt(self.folder + file_name, add_to_table, fmt='%.3e')
                    else:
                        np.savetxt(self.folder + file_name, tab, fmt='%.3e')
            else:
                if self.point_like:
                    rb_med = np.log10(10. ** (-4.24) * m ** 0.459)
                    rb_low = rb_med - 1.
                    rb_high = rb_med + 1.
                    rb_list = np.logspace(rb_low, rb_high, 20)
                    gamma_list = np.linspace(0., 1.45, 20)
                else:
                    rb_med = np.log10(10. ** (-4.24) * m ** 0.459)
                    rb_low = rb_med - .75
                    rb_high = rb_med + .75
                    rb_list = np.logspace(rb_low, rb_high, 10)
                    gamma_list = np.linspace(0, 1.45, 8)
                temp_arry = np.zeros(rb_list.size * len(gamma_list))
                jcnt = 0
                for rb in rb_list:
                    print '    Rb Parameter: ', rb
                    for j, gam in enumerate(gamma_list):
                        print '         Gamma: ', gam
                        try:
                            look_array = np.loadtxt(self.folder + file_name)
                            mlook = float('{:.3e}'.format(m))
                            rblook = float('{:.3e}'.format(rb))
                            gamlook = float('{:.3e}'.format(gam))
                            if np.sum((mlook == look_array[:, 0]) & (rblook == look_array[:, 1]) &
                                              (gamlook == look_array[:, 2])) == 1:
                                exists = True
                            else:
                                exists = False
                        except:
                            exists = False
                        if not exists:
                            dm_model = Model(self.mx, self.cross_sec, self.annih_prod,
                                             m, profile=self.profile, pointlike=self.point_like,
                                             m200=self.m200, stiff_rb=self.stiff_rb, gam=gam,
                                             rb=rb)
                            if self.point_like:
                                temp_arry[jcnt] = dm_model.d_max_point(threshold=threshold) * \
                                                  self.hw_prob_gamma(gam) * self.hw_prob_rb(rb, m)
                            else:
                                dmx = dm_model.D_max_extend()
                                print '             dmax: ', dmx
                                temp_arry[jcnt] = dmx
                                #temp_arry[jcnt] = dmx * self.hw_prob_gamma(gam) * self.hw_prob_rb(rb, m)
                            jcnt += 1
                    #pre_marg = np.reshape(temp_arry, (rb_list.size, len(gamma_list)))
                    #dmax = RectBivariateSpline(rb_list, gamma_list,
                    #                           pre_marg).integral(np.min(rb_list),
                    #                                              np.max(rb_list), 0.,
                    #                                              np.max(gamma_list))

                            tab = np.array([m, rb, gam, temp_arry[jcnt - 1]]).transpose()
                            if os.path.isfile(self.folder+file_name):
                                load_info = np.loadtxt(self.folder + file_name)
                                add_to_table = np.vstack((load_info, tab))
                                np.savetxt(self.folder + file_name, add_to_table, fmt='%.3e')
                            else:
                                np.savetxt(self.folder + file_name, tab, fmt='%.3e')
        return


    def N_obs(self, bmin, plike=True):
        """
        For pre tabled d_max functions, calculates the number of observable subhalos

        :param bmin: These analyses cut out the galactic plane, b_min (in degrees) specifies location
        of the cut
        """

        def prob_c(c, m):
            cm = Concentration_parameter(m, arxiv_num=self.arxiv_num)
            #sigma_c = 0.24 * np.log(10.)
            sigma_c = 0.14 * np.log(10.)
            return (np.exp(- (np.log(c / cm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                    (np.sqrt(2. * np.pi) * sigma_c * c))
        if plike:
            file_name = 'Dmax_POINTLIKE_' + str(Profile_list[self.profile]) + self.tr_tag + \
                        '_mx_' + str(self.mx) + '_cross_sec_{:.3e}'.format(self.cross_sec) + \
                        '_annih_prod_' + self.annih_prod + self.extra_tag + '.dat'
        else:
            file_name = 'Dmax_Extended_' + str(Profile_list[self.profile]) + self.tr_tag + \
                        '_mx_' + str(self.mx) + '_cross_sec_{:.3e}'.format(self.cross_sec) + \
                        '_annih_prod_' + self.annih_prod + self.extra_tag + '.dat'

        if self.profile == 0:
            integrand_table = np.loadtxt(self.folder + file_name)
            mass_list = np.unique(integrand_table[:, 0])
            c_list = np.unique(integrand_table[:, 1])

            if self.truncate:
                divis = 0.005
            else:
                divis = 1.

            m_num = mass_list.size
            c_num = c_list.size

            integrand_table[:, 2] = (260. * (integrand_table[:, 0]) ** (-1.9) *
                                     prob_c(integrand_table[:, 1], integrand_table[:, 0] / divis) *
                                     (integrand_table[:, 2] ** 3.) / 3.0)

            int_prep_spline = np.reshape(integrand_table[:, 2], (m_num, c_num))
            integrand = RectBivariateSpline(mass_list, c_list, int_prep_spline)

            integr = integrand.integral(np.min(mass_list), np.max(mass_list), np.min(c_list), np.max(c_list))
            print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
            return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr
        elif self.profile == 1:
            integrand_table = np.loadtxt(self.folder + file_name)
            mass_list = np.unique(integrand_table[:, 0])

            integrand_table[:, 1] = (260. * (integrand_table[:, 0]) ** (-1.9) *
                                     integrand_table[:, 1] ** 3. / 3.0)

            integrand_interp = interp1d(mass_list, integrand_table[:, 1], kind='linear')
            mass_full = np.logspace(np.log10(np.min(mass_list)), np.log10(np.max(mass_list)), 200)
            integr = np.trapz(integrand_interp(mass_full), mass_full)
            print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
            return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr
        elif self.profile == 2:
            integrand_table = np.loadtxt(self.folder + file_name)
            mass_list = np.unique(integrand_table[:, 0])

            integrand_table[:, 1] = (628. * (integrand_table[:, 0]) ** (-1.9) *
                                     (integrand_table[:, 1] ** 3.) / 3.0)
            integrand_interp = interp1d(mass_list, integrand_table[:,1], kind='linear')
            mass_full = np.logspace(np.log10(np.min(mass_list)), np.log10(np.max(mass_list)), 200)
            integr = np.trapz(integrand_interp(mass_full), mass_full)
            #integrand = UnivariateSpline(mass_list, integrand_table[:, 1])
            #integr = integrand.integral(np.min(mass_list), np.max(mass_list))
            print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
            return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr

    def hw_prob_rb(self, rb, mass):
        rb_norm = 10. ** (-4.24) * mass ** 0.459
        sigma_c = 0.47
        return (np.exp(- (np.log(rb / rb_norm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                (np.sqrt(2. * np.pi) * sigma_c * rb))

    def hw_prob_gamma(self, gam):
        # norm inserted b/c integration truncated on region [0, 1.45]
        sigma = 0.426
        k = 0.1
        mu = 0.85
        y = -1. / k * np.log(1. - k * (gam - mu) / sigma)
        norm = quad(lambda x: np.exp(- x ** 2. / 2.) / (np.sqrt(2. * np.pi) * (sigma - k * (gam - mu))), 0., 1.45)[0]
        return np.exp(- y ** 2. / 2.) / (np.sqrt(2. * np.pi) * (sigma - k * (gam - mu))) / norm
