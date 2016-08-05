# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from helper import *
import helper
from Limits import *
from profiles import *
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import RectBivariateSpline,interp1d,interp2d
from scipy.optimize import minimize,fminbound,brentq
import os
import pickle
import glob
from Plotting_Functions import *
import time


try:
    MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'


class Model(object):
    """NOT READY"""

    def __init__(self, mx, cross_sec, annih_prod, halo_mass, alpha, 
                 concentration_param=None, z=0., truncate=False,
                 arxiv_num=10070438, profile=0, pointlike=True):

        self.mx = mx
        self.c_sec = cross_sec
        self.annih_prod = annih_prod
        self.arxiv_num = arxiv_num
        self.truncate = truncate
        self.alpha = alpha
        self.profile = profile
        self.plike = str2bool(pointlike)
        if not self.plike:
            self.gamma = self.Determine_Gamma()
        
        if profile == 0:
            self.subhalo = Einasto(halo_mass, alpha, 
                                   concentration_param=concentration_param, 
                                   truncate=truncate, arxiv_num=arxiv_num)
        elif profile == 1:
            self.subhalo = NFW(halo_mass, alpha, 
                               concentration_param=concentration_param, 
                               truncate=truncate, arxiv_num=arxiv_num)
        else:
            "Profile does not exist..."  
            exit()                   

    def min_Flux(self, dist):
        extension = self.subhalo.Spatial_Extension(dist)
        return self.Threshold(self.gamma, extension)

    def Total_Flux(self, dist):
        pre_factor = self.c_sec / (8. * np.pi * self.mx**2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
            self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1], kind='cubic')
        n_gamma = integrated_rate(self.mx)
        flux = pre_factor * n_gamma * 10**self.subhalo.J(dist, 90.)
        return flux
        
    def Partial_Flux(self, dist, theta):
        pre_factor = self.c_sec / (8. * np.pi * self.mx**2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
            self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:,0],integrated_list[:,1],kind='cubic')
        
        n_gamma = integrated_rate(self.mx)
        
        flux = pre_factor * n_gamma * 10**self.subhalo.J(dist, theta)
        return flux

    def d_max_point(self, threshold=(7.0 * 10 ** (-10))):
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

        def flux_diff_lten(x):
            return np.abs(self.Total_Flux(10**x) - self.min_Flux(10**x))
        d_max = fminbound(flux_diff_lten, -4., 1.4, xtol= 10**-3.)
        print 10.**float(d_max)
        return 10.**float(d_max)

    def Determine_Gamma(self):
        #  TODO: Generalize beyond b\bar{b}
        #  Also carefully check the accuracy
        num_collisions = 10 ** 5.
        spec_file = MAIN_PATH + "/Spectrum/" + str(int(self.mx)) + "GeVDMspectrum.dat"
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + self.annih_prod + ".dat"

        spectrum = np.loadtxt(spec_file)
        spectrum[:, 1] = spectrum[:, 1] / num_collisions

        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1])

        gamma_list = [1.5, 2.0, 2.5, 3.0]
        normalize = np.zeros(len(gamma_list))
        for g in range(len(gamma_list)):
            normalize[g] = integrated_rate(self.mx) / integrate.quad(lambda x: x ** (-gamma_list[g]), 1., self.mx)[0]

        int_meanes = lambda x: interpola(x, spectrum[:, 0], spectrum[:, 0] * spectrum[:, 1])
        meanE = integrate.quad(int_meanes, 1., self.mx, epsabs=10.**-4, epsrel=10.**-4)[0]

        meanE_gam = np.zeros(len(gamma_list))
        for g in range(len(gamma_list)):
            meanE_gam[g] = integrate.quad(lambda x: normalize[g] * x ** (-gamma_list[g] + 1.), 1., self.mx,
                                          epsabs=10.**-4, epsrel=10.**-4)[0]

        diff_meanE = np.abs(meanE_gam - meanE)

        return gamma_list[diff_meanE.argmin()]

    def Threshold(self, gamma, extension):

        file = MAIN_PATH + "/ExtendedThresholds/DetectionThresholdGamma" + \
               str(int(gamma * 10)) + ".dat"

        thresh_list = np.loadtxt(file)
        if extension > 2.0:
            return interpola(2.0, thresh_list[:, 0], thresh_list[:, 1])
        else:
            return interpola(extension, thresh_list[:, 0], thresh_list[:, 1])


class Observable(object):
    
    def __init__(self, mx, cross_sec, annih_prod, m_low=np.log10(3.24 * 10**4.),
                 m_high=np.log10(1.0 * 10 ** 7.), c_low=np.log10(20.),
                 c_high=2.1, alpha=0.16, profile=0, truncate=False, 
                 arxiv_num=10070438, point_like=True):

        point_like = str2bool(point_like)
        Profile_list = ["Einasto", "NFW"]
        self.truncate = truncate
        self.mx = mx
        self.cross_sec = cross_sec
        self.annih_prod = annih_prod
        self.c_low = c_low
        self.c_high = c_high
        self.alpha = alpha
        self.profile = profile
        self.profile_name = Profile_list[self.profile]
        if self.truncate:
            self.m_low = np.log10(10.**m_low / 0.005)
            self.m_high = np.log10(10.**m_high / 0.005)
        else:
            self.m_low = m_low
            self.m_high = m_high
        self.point_like = point_like
        if self.point_like:
            ptag = '_Pointlike'
        else:
            ptag = '_Extended'

        self.arxiv_num = arxiv_num
        
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"
        info_str = "Observable_Profile_" + self.profile_name + "_Truncate_" +\
            str(self.truncate) + ptag + "_mx_" + str(mx) + "_annih_prod_" +\
            annih_prod + "_arxiv_num_" + str(arxiv_num) + "/"
        
        self.folder += info_str
        ensure_dir(self.folder)
                   
        default_dict = {'profile': 'Einasto', 'truncate': False, 'mx': 100., 'alpha': 0.16,
                        'annih_prod': 'BB', 'arxiv_num': 10070438, 'c_low': np.log10(20.),
                        'c_high': 2.1, 'c_num': 15, 'm_low': np.log10(3.24 * 10**4.),
                        'm_high': np.log10(1.0 * 10 **7), 'm_num': 30}
        self.param_list = default_dict
        self.param_list['profile'] = self.profile_name
        self.param_list['truncate'] = self.truncate
        self.param_list['mx'] = self.mx
        self.param_list['annih_prod'] = self.annih_prod
        self.param_list['arxiv_num'] = self.arxiv_num
        self.param_list['c_low'] = self.c_low
        self.param_list['c_high'] = self.c_high
        self.param_list['m_low'] = self.m_low 
        self.param_list['m_hgih'] = self.m_high 

    def Table_Spatial_Extension(self, d_low=-3., d_high=1., d_num=80, m_num=60, 
                                c_num=50):
        """ Tables spatial extension for future use. 
            
            Profile Numbers correspond to [Einasto, NFW] # 0 - 1 (TODO: add NFW)
        """
        Profile_names=['Einasto','NFW']

        file_name = 'SpatialExtension_' + str(Profile_names[self.profile]) + '_Truncate_' +\
                    str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
                    str(self.alpha) + '.dat'
                        
        mass_list = np.logspace(self.m_low, self.m_high, m_num)
        dist_list = np.logspace(d_low, d_high, d_num)
        c_list = np.logspace(self.c_low, self.c_high, c_num)  
        
        for m in mass_list:
            print 'Subhalo mass: ', m
            for c in c_list:
                print 'Concentration parameter: ', c
                
                extension_tab = np.zeros(len(dist_list))
                if self.profile == 0:
                    subhalo = Einasto(m, self.alpha, c, truncate=self.truncate, 
                                      arxiv_num=self.arxiv_num)
                elif self.profile == 1:
                    subhalo = NFW(m, self.alpha, c, truncate=self.truncate, 
                                  arxiv_num=self.arxiv_num)
                else:
                    'Profile not implimented yet'
                
                for ind,d in enumerate(dist_list):
                    try:
                        look_array = np.loadtxt(self.folder + file_name)
                        if any((np.round([m,c,d],5) == x).all() for x in np.round(look_array[:,0:3],5)):
                            exists = True
                        else:
                            exists = False
                    except:
                        exists = False
                    
                    if not exists and subhalo.Full_Extension(d) > .05:
                        extension_tab[ind] = subhalo.Spatial_Extension(d)
    
                extension_tab = extension_tab[np.nonzero(extension_tab)]
                assert isinstance(extension_tab, object)
                entries_added = extension_tab.size
                full_tab = np.vstack((np.ones(entries_added) * m,
                                      np.ones(entries_added) * c, 
                                      dist_list[:entries_added],
                                      extension_tab)).transpose()
                    
                if os.path.isfile(self.folder+file_name):
                    load_info = np.loadtxt(self.folder + file_name)
                    add_to_table = np.vstack((load_info,full_tab))
                    np.savetxt(self.folder + file_name, add_to_table)
                else:
                    np.savetxt(self.folder + file_name, full_tab)
        return
    
    def Table_Dmax_Pointlike(self, m_num=20, c_num = 15, threshold=1.*10.**-9.):
        """ Tables d_max extended for future use. 

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
        
        Profile_names=['Einasto','NFW']
        
        file_name = 'Dmax_POINTLIKE_' + str(Profile_names[self.profile]) + '_Truncate_' +\
                    str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
                    str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' +\
                    str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'
                        
        mass_list = np.logspace(self.m_low, self.m_high, m_num)
        c_list = np.logspace(self.c_low, self.c_high, c_num)  
        
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
                                     profile=self.profile, point_like=self.point_like)
                    
                    dmax = dm_model.d_max_point(threshold=threshold)
                    if self.truncate:
                        mm = 0.005 * m
                    else:
                        mm = m
                    tab = np.array([mm, c, dmax]).transpose()
                        
                    if os.path.isfile(self.folder+file_name):
                        load_info = np.loadtxt(self.folder + file_name)
                        add_to_table = np.vstack((load_info,tab))
                        np.savetxt(self.folder + file_name, add_to_table)
                    else:
                        np.savetxt(self.folder + file_name, tab)        
        return

    def Table_Dmax_Extended(self, m_num=20, c_num=15):
        """ Tables d_max extended for future use.

        """
        self.param_list['m_num'] = m_num
        self.param_list['c_num'] = c_num

        if os.path.isfile(self.folder + "param_list.pkl"):
            openfile = open(self.folder + "param_list.pkl", 'rb')
            old_dict = pickle.load(openfile)
            openfile.close()
            check_diff = DictDiffer(self.param_list, old_dict)
            if bool(check_diff.changed()):
                [os.remove(f) for f in os.listdir(self.folder)]
                pickle.dump(self.param_list, open(self.folder + "param_list.pkl", "wb"))
        else:
            pickle.dump(self.param_list, open(self.folder + "param_list.pkl", "wb"))

        Profile_names = ['Einasto', 'NFW']

        file_name = 'Dmax_' + str(Profile_names[self.profile]) + '_Truncate_' + \
                    str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' + \
                    str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' + \
                    str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'

        mass_list = np.logspace(self.m_low, self.m_high, m_num)
        c_list = np.logspace(self.c_low, self.c_high, c_num)

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
                                     profile=self.profile, pointlike=self.point_like)

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
    
    def N_Extended(self, bmin):
            
        Profile_names=['Einasto','NFW']
        
#        openfile = open(self.folder+"param_list.pkl", 'rb')
#        dict = pickle.load(openfile)
#        openfile.close()       
            
        def prob_c(c, M):
            cm = Concentration_parameter(M, arxiv_num=self.arxiv_num)
            sigma_c = 0.24
            return (np.exp(- (np.log(c / cm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                   (np.sqrt(2. * np.pi) * sigma_c * c))
           
        file_name = 'Dmax_Extended' + str(Profile_names[self.profile]) + '_Truncate_' +\
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
            if not np.any(np.in1d(mass_list, round(integrand_table[i,0],5))):
                mass_list = np.append(mass_list, round(integrand_table[i,0],5))
            if not np.any(np.in1d(c_list, round(integrand_table[i,1],5))):
                c_list = np.append(c_list, round(integrand_table[i,1],5))
            
        m_num = mass_list.size
        c_num = c_list.size
        
        int_prep_spline = np.reshape(integrand_table[:,2], (m_num, c_num))
       
        integrand = RectBivariateSpline(mass_list, c_list, int_prep_spline)
        integr = integrand.integral(3.24 * 10**4., 1.0 * 10**7., 0., 1000.)
                                    
        # TODO: consider alternative ways of performing this integral
    
        return 2. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr

    def N_Pointlike(self, bmin):

        Profile_names = ['Einasto', 'NFW']

        openfile = open(self.folder+"param_list.pkl", 'rb')
        dict = pickle.load(openfile)
        openfile.close()
        mass_list = np.logspace(dict['m_low'], dict['m_high'], dict['m_num'])
        c_list = np.logspace(dict['c_low'], dict['c_high'], dict['c_num'])

        def prob_c(c, M):
            cm = Concentration_parameter(M, arxiv_num=self.arxiv_num)
            sigma_c = 0.24
            return (np.exp(- (np.log(c / cm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                    (np.sqrt(2. * np.pi) * sigma_c * c))

        file_name = 'Dmax_POINTLIKE_' + str(Profile_names[self.profile]) + '_Truncate_' + \
                    str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' + \
                    str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' + \
                    str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'

        integrand_table = np.loadtxt(self.folder + file_name)
        if self.truncate:
            integrand_table[:, 2] = (260. * (integrand_table[:, 0])**(-1.9) *
                                     prob_c(integrand_table[:, 1], integrand_table[:, 0] / 0.005) *
                                     integrand_table[:, 2]**3. / 3.0)
        else:
            integrand_table[:, 2] = (260. * (integrand_table[:, 0]) ** (-1.9) *
                                     prob_c(integrand_table[:, 1], integrand_table[:, 0]) *
                                     integrand_table[:, 2] ** 3. / 3.0)

        m_num = mass_list.size
        c_num = c_list.size

        int_prep_spline = np.reshape(integrand_table[:, 2], (m_num, c_num))

        integrand = RectBivariateSpline(mass_list, c_list, int_prep_spline)
        integr = integrand.integral(3.24 * 10. ** 4., 10. ** 7., 0., np.inf)

        print self.cross_sec, (4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr)
        return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr


