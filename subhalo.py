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
from scipy.interpolate import RectBivariateSpline,interp1d,interp2d
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
                 arxiv_num=10070438, profile=0, pointlike=False,
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
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1], kind='cubic')
        n_gamma = integrated_rate(self.mx)

        jf = integrate.quad(lambda x: 4. * np.pi * kpctocm * self.subhalo.density(x) ** 2. * x ** 2.,
                            0., float(self.subhalo.max_radius), epsabs=10**-5., epsrel=10**-5.)
        return np.sqrt(pre_factor * n_gamma * jf[0] / threshold)

    def D_max_extend(self):
        """
        Calculates the maximum distance a spatially extended subhalo can be to still be observable
        :return: distance in kpc
        """
        def flux_diff_lten(x):
            return np.abs(self.Total_Flux(10. ** x) - self.min_Flux(10. ** x))
        d_max = fminbound(flux_diff_lten, -4., 2., xtol= 10**-4.)
        return 10.**float(d_max)

    def Determine_Gamma(self):
        """
        Caclualtes the spectral index to be used to determine the spatial extended threshold
        :return: [1.5, 2.0, 2.5, 3.0] whichever spectral index most closely produces same average
        photon energy
        """
        #  TODO: Generalize beyond b\bar{b}
        #  TODO: Derive this information from Pythia8 (currently done using PPPC)
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
                 m_high=np.log10(1.0 * 10 ** 7.), c_low=np.log10(20.),
                 c_high=2.1, alpha=0.16, profile=0, truncate=False, 
                 arxiv_num=10070438, point_like=True, m200=False,
                 gam=0.88, stiff_rb=False):

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

        self.arxiv_num = arxiv_num
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"
        info_str = "Observable_Profile_" + self.profile_name + self.tr_tag +\
            ptag + "_mx_" + str(mx) + "_annih_prod_" +\
            annih_prod + self.extra_tag + '_Mlow_{:.3e}'.format(m_low) + "/"
        
        self.folder += info_str
        ensure_dir(self.folder)
                   
        default_dict = {'profile': 'Einasto', 'truncate': False, 'mx': 100., 'alpha': 0.16,
                        'annih_prod': 'BB', 'arxiv_num': 10070438, 'c_low': np.log10(20.),
                        'c_high': 2.1, 'c_num': 15, 'm_low': np.log10(3.24 * 10 ** 4.),
                        'm_high': np.log10(1.0 * 10 ** 7), 'm_num': 30, 'gamma': 0.945,
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

    
    def Table_Dmax_Pointlike(self, m_num=20, c_num=15, threshold=7.*10.**-10.):
        """
        Tables the maximimum distance a point-like source can be detected for a specified
        threshold given a particular DM candidate and subhalo specificiations
        """
        self.param_list['m_num'] = m_num
        self.param_list['c_num'] = c_num
        
        if os.path.isfile(self.folder+"param_list.pkl"):
            openfile = open(self.folder+"param_list.pkl", 'rb')
            old_dict = pickle.load(openfile)
            openfile.close()
            check_diff = DictDiffer(self.param_list, old_dict)
            if bool(check_diff.changed()):
                files = glob.glob(self.folder + '/*')
                for f in files:
                    os.remove(f)
                pickle.dump(self.param_list, open(self.folder+"param_list.pkl", "wb")) 
        else:
            pickle.dump(self.param_list, open(self.folder+"param_list.pkl", "wb"))
        
        file_name = 'Dmax_POINTLIKE_' + str(Profile_list[self.profile]) + self.tr_tag +\
                    '_mx_' + str(self.mx) + '_cross_sec_{:.3e}'.format(self.cross_sec) +\
                    '_annih_prod_' + self.annih_prod + self.extra_tag + '.dat'
        mass_list = np.logspace(self.m_low, self.m_high, (self.m_high - self.m_low) * 6)

        print 'Cross Section: ', self.cross_sec, '\n'
        for m in mass_list:
            print 'Subhalo mass: ', m
            if self.profile < 2:
                c_list = np.logspace(self.c_low, self.c_high, c_num)
                for c in c_list:
                    print '    Concentration parameter: ', c
                    try:
                        look_array = np.loadtxt(self.folder + file_name)
                        if self.truncate:
                            mm = 0.005 * m
                        else:
                            mm = m
                        if any((np.round([mm,c],4) == x).all() for x in np.round(look_array[:,0:2],4)):
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

                        dmax = dm_model.d_max_point(threshold=threshold)
                        if self.truncate:
                            mm = 0.005 * m
                        else:
                            mm = m
                        tab = np.array([mm, c, dmax]).transpose()

                        if os.path.isfile(self.folder+file_name):
                            load_info = np.loadtxt(self.folder + file_name)
                            add_to_table = np.vstack((load_info, tab))
                            np.savetxt(self.folder + file_name, add_to_table)
                        else:
                            np.savetxt(self.folder + file_name, tab)
            else:

                rb_list = np.logspace(-3, np.log10(0.5), 20)
                gamma_list = np.linspace(0.2, 0.85 + 0.351 / 0.861 - 0.1, 20)
                for rb in rb_list:
                    print '    Rb Parameter: ', rb
                    temp_arry = np.zeros(len(gamma_list) * 2).reshape(len(gamma_list), 2)
                    for j, gam in enumerate(gamma_list):
                        print '         Gamma: ', gam
                        dm_model = Model(self.mx, self.cross_sec, self.annih_prod,
                                         m, profile=self.profile, pointlike=self.point_like,
                                         m200=self.m200, stiff_rb=self.stiff_rb, gam=gam,
                                         rb=rb)

                        temp_arry[j] = [gam, dm_model.d_max_point(threshold=threshold)]
                    dmax = UnivariateSpline(temp_arry[:, 0], temp_arry[:, 1] *
                                            self.hw_prob_gamma(gam)).integral(0., 0.85 + 0.351 / 0.861)

                    tab = np.array([m, rb, dmax]).transpose()
                    if os.path.isfile(self.folder+file_name):
                        load_info = np.loadtxt(self.folder + file_name)
                        add_to_table = np.vstack((load_info, tab))
                        np.savetxt(self.folder + file_name, add_to_table, fmt='%.3e')
                    else:
                        np.savetxt(self.folder + file_name, tab, fmt='%.3e')
        return

    def Table_Dmax_Extended(self, m_num=20, c_num=15):
        """
        Tables the maximimum distance an extended source can be detected for a specified
        threshold given a particular DM candidate and subhalo specificiations
        """
        self.param_list['m_num'] = m_num
        self.param_list['c_num'] = c_num

        if os.path.isfile(self.folder + "param_list.pkl"):
            openfile = open(self.folder + "param_list.pkl", 'rb')
            old_dict = pickle.load(openfile)
            openfile.close()
            check_diff = DictDiffer(self.param_list, old_dict)
            if bool(check_diff.changed()):
                files = glob.glob(self.folder + '/*')
                for f in files:
                    os.remove(f)
                pickle.dump(self.param_list, open(self.folder + "param_list.pkl", "wb"))
        else:
            pickle.dump(self.param_list, open(self.folder + "param_list.pkl", "wb"))

        file_name = 'Dmax_' + str(Profile_list[self.profile]) + '_mx_' + str(self.mx) +\
                    '_cross_sec_{:.3e}'.format(self.cross_sec) + '_annih_prod_' + self.annih_prod + '.dat'

        mass_list = np.logspace(self.m_low, self.m_high, m_num)
        c_list = np.logspace(self.c_low, self.c_high, c_num)
        print 'Cross Section: ', self.cross_sec, '\n'
        for m in mass_list:
            print 'Subhalo mass: ', m
            for c in c_list:
                print '    Concentration parameter: ', c
                try:
                    look_array = np.loadtxt(self.folder + file_name)
                    if self.truncate:
                        mm = 0.005 * m
                    else:
                        mm = m
                    if any((np.round([mm, c], 4) == x).all() for x in np.round(look_array[:, 0:2], 4)):
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

                    dmax = dm_model.D_max_extend()
                    if self.truncate:
                        mm = 0.005 * m
                    else:
                        mm = m
                    tab = np.array([mm, c, dmax]).transpose()

                    if os.path.isfile(self.folder + file_name):
                        load_info = np.loadtxt(self.folder + file_name)
                        add_to_table = np.vstack((load_info, tab))
                        np.savetxt(self.folder + file_name, add_to_table)
                    else:
                        np.savetxt(self.folder + file_name, tab)
        return

    def Table_Dmax_Extended_Constrained(self, min_extension=0.3):
        """
        Tables the maximum distance a subhalo to be observed as spatially extended
        and have a specified minimum extension (in degrees)

        Note: Table_Dmax_Extended should have already been run!
        """
        # TODO: Test. Also think about whether min_ext should be compared to full extension or that calulated from 68% containment
        Profile_names = ['Einasto', 'NFW']

        load_file_name = 'Dmax_' + str(Profile_names[self.profile]) + '_Truncate_' + \
            str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' + \
            str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' + \
            str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'

        save_file_name = 'Dmax_Constrained_MinExtension_' + str(min_extension) +\
            str(Profile_names[self.profile]) + '_Truncate_' + \
            str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' + \
            str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' + \
            str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'

        try:
            load_dmax = np.loadtxt(self.folder + load_file_name)
        except:
            load_dmax = []
            print 'File Not Found! Exiting...'
            exit()

        print 'Cross Section: ', self.cross_sec, '\n'

        def extension_line(d, model):
            return np.abs(model.subhalo.Spatial_Extension(d) - min_extension)

        for i in range(load_dmax[:, 0]):
            m = load_dmax[i, 0]
            c = load_dmax[i, 1]
            print 'Subhalo mass: ', m, 'Concentration parameter: ', c

            dm_model = Model(self.mx, self.cross_sec, self.annih_prod,
                             m, self.alpha, concentration_param=c,
                             truncate=self.truncate,
                             arxiv_num=self.arxiv_num,
                             profile=self.profile, pointlike=False,
                             m200=self.m200)

            dext = fminbound(extension_line, -5., np.log10(load_dmax[i, 2]),
                             args=dm_model)
            if self.truncate:
                mm = 0.005 * m
            else:
                mm = m

            tab = np.array([mm, c, np.power(10., dext)]).transpose()

            if os.path.isfile(self.folder + save_file_name):
                load_info = np.loadtxt(self.folder + save_file_name)
                add_to_table = np.vstack((load_info, tab))
                np.savetxt(self.folder + save_file_name, add_to_table)
            else:
                np.savetxt(self.folder + save_file_name, tab)
        return

    def N_Extended(self, bmin, constrained=False, min_extension=0.3):
        """
        For pre tabled d_max functions, calculates the number of observable spatially
        extended subhalos

        :param bmin: These anlayses cut out the galactic plane, b_min (in degrees) specifies location
        of the cut
        """
        
#        openfile = open(self.folder+"param_list.pkl", 'rb')
#        dict = pickle.load(openfile)
#        openfile.close()       
            
        def prob_c(c, M):
            cm = Concentration_parameter(M, arxiv_num=self.arxiv_num)
            sigma_c = 0.24
            return (np.exp(- (np.log(c / cm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                   (np.sqrt(2. * np.pi) * sigma_c * c))

        if constrained:
            file_name = 'Dmax__Constrained_MinExtension_' + str(min_extension) +\
                str(Profile_list[self.profile]) + '_Truncate_' +\
                str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
                str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' +\
                str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'
        else:
            file_name = 'Dmax_' + str(Profile_list[self.profile]) + '_Truncate_' +\
                str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
                str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' +\
                str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'

        integrand_table = np.loadtxt(self.folder+file_name)
        if self.truncate:
            integrand_table[:, 2] = (260. * (integrand_table[:, 0]) ** (-1.9) *
                                     prob_c(integrand_table[:, 1], integrand_table[:, 0] / 0.005) *
                                     integrand_table[:, 2] ** 3. / 3.0)
        else:
            integrand_table[:, 2] = (260. * (integrand_table[:, 0]) ** (-1.9) *
                                     prob_c(integrand_table[:, 1], integrand_table[:, 0]) *
                                     integrand_table[:, 2] ** 3. / 3.0)
                                
        mass_list = np.array([round(integrand_table[0,0],5)])
        c_list = np.array([round(integrand_table[0,1],5)])
        for i in range(integrand_table[:,0].size):
            if not np.any([np.in1d(mass_list, [round(integrand_table[i, 0], 5)])]):
                mass_list = np.append(mass_list, [round(integrand_table[i,0], 5)])
            if not np.any([np.in1d(c_list, [round(integrand_table[i, 1], 5)])]):
                c_list = np.append(c_list, [round(integrand_table[i, 1], 5)])
            
        m_num = mass_list.size
        c_num = c_list.size
        int_prep_spline = np.reshape(integrand_table[:,2], (m_num, c_num))
        integrand = RectBivariateSpline(mass_list, c_list, int_prep_spline)
        integr = integrand.integral(3.24 * 10**4., 1.0 * 10**7., 0., 300.)

        print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
        return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr

    def N_Pointlike(self, bmin):
        """
        For pre tabled d_max functions, calculates the number of observable point-like subhalos

        :param bmin: These analyses cut out the galactic plane, b_min (in degrees) specifies location
        of the cut
        """
        #openfile = open(self.folder+"param_list.pkl", 'rb')
        #dict = pickle.load(openfile)
        #openfile.close()

        def prob_c(c, m):
            cm = Concentration_parameter(m, arxiv_num=self.arxiv_num)
            sigma_c = 0.24
            return (np.exp(- (np.log(c / cm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                    (np.sqrt(2. * np.pi) * sigma_c * c))


        file_name = 'Dmax_POINTLIKE_' + str(Profile_list[self.profile]) + self.tr_tag + \
                    '_mx_' + str(self.mx) + '_cross_sec_{:.3e}'.format(self.cross_sec) + \
                    '_annih_prod_' + self.annih_prod + self.extra_tag + '.dat'

        if self.profile < 2:
            integrand_table = np.loadtxt(self.folder + file_name)
            mass_list = np.unique(integrand_table[:, 0])
            c_list = np.unique(integrand_table[:, 1])
            if self.truncate:
                divis = 0.005
            else:
                divis = 1.
            integrand_table[:, 2] = (260. * (integrand_table[:, 0])**(-1.9) *
                                     prob_c(integrand_table[:, 1], integrand_table[:, 0] / divis) *
                                     integrand_table[:, 2]**3. / 3.0)
            m_num = mass_list.size
            c_num = c_list.size
            int_prep_spline = np.reshape(integrand_table[:, 2], (m_num, c_num))
            integrand = RectBivariateSpline(mass_list, c_list, int_prep_spline)
            integr = integrand.integral(np.min(mass_list), np.max(mass_list),
                                        10. ** -4., 1.)
            print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
            return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr
        else:
            integrand_table = np.loadtxt(self.folder + file_name)

            mass_list = np.unique(integrand_table[:, 0])
            rb_list = np.unique(integrand_table[:, 1])

            integrand_table[:, 2] = (110. * (integrand_table[:, 0]) ** (-1.9) *
                                     self.hw_prob_rb(integrand_table[:, 1], integrand_table[:, 0]) *
                                     integrand_table[:, 2] ** 3. / 3.0)

            m_num = mass_list.size
            rb_num = rb_list.size

            int_prep_spline = np.reshape(integrand_table[:, 2], (m_num, rb_num))

            integrand = RectBivariateSpline(mass_list, rb_list, int_prep_spline)
            integr = integrand.integral(np.min(mass_list), np.max(mass_list),
                                        10.**-4., 1.)
            print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
            return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr

    def hw_prob_rb(self, rb, mass):
        rb_norm = 10. ** (-4.24) * mass ** 0.459
        sigma_c = 0.47
        return (np.exp(- (np.log(rb / rb_norm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                (np.sqrt(2. * np.pi) * sigma_c * rb))

    def hw_prob_gamma(self, gam):
        sigma = 0.426
        k = 0.1
        mu = 0.85
        y = -1. / k * np.log(1. - k * (gam - mu) / sigma)
        return np.exp(- y ** 2. / 2.) / (np.sqrt(2. * np.pi) * (sigma - k * (gam - mu)))
