# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:46:43 2016

@author: SamWitte
"""

import numpy as np
from helper import *
from Threshold import *
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import RectBivariateSpline,interp1d
from scipy.optimize import minimize,minimize_scalar,brute
import os
import pickle
import glob

MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']

class Model(object):
    """NOT READY"""
    def __init__(self, mx, cross_sec, annih_prod, halo_mass, alpha, 
                 concentration_param=None, z=0., truncate=False, 
                 arxiv_num=10070438, profile=0):
        self.mx = mx
        self.c_sec = cross_sec
        self.annih_prod = annih_prod
        self.arxiv_num = arxiv_num
        self.truncate = truncate
        self.alpha = alpha
        self.profile = profile        
        
        if profile == 0:
            self.subhalo = Einasto(halo_mass, alpha, 
                                   concentration_param=concentration_param, 
                                   truncate=truncate, arxiv_num=arxiv_num)
        elif profile ==1:
            self.subhalo = NFW(halo_mass, alpha, 
                               concentration_param=concentration_param, 
                               truncate=truncate, arxiv_num=arxiv_num)
        else:
            "Profile does not exist..."  
            exit()                   

    def min_Flux(self, dist):
        gamma_index = Determine_Gamma(self.mx, self.annih_prod)       
        extension = 2.0*self.subhalo.Spatial_Extension(dist)       
        
        return Threshold(gamma_index, extension)
        
        
    def Total_Flux(self, dist):
        prefactor = self.c_sec / (8. * np.pi * self.mx**2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
                         self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:,0],integrated_list[:,1])        
        n_gamma = integrated_rate(self.mx)
        flux = prefactor * n_gamma * 10**self.subhalo.J_pointlike(dist)
        return flux
        
    def Partial_Flux(self, dist, theta):
        prefactor = self.c_sec / (8. * np.pi * self.mx**2.)
        integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + \
                         self.annih_prod + ".dat"
        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:,0],integrated_list[:,1])
        
        n_gamma = integrated_rate(self.mx)
        
        flux = prefactor * n_gamma * 10**self.subhalo.J(dist, theta)
        return flux

    def D_max_extend(self):
        
        def flux_diff(x):
            return np.abs(np.log10(self.Total_Flux(x)) - np.log10(self.min_Flux(x)))
        if self.profile == 0:
            dist_tab = np.logspace(-1.5, 1.4, 40)
        elif self.profile == 1:
            dist_tab = np.logspace(-1.5, 1.4, 40)
        
        f_difftab = np.zeros((dist_tab.size,2))

        for i,dist in enumerate(dist_tab):
            try:
                f_difftab[i,1] = flux_diff(dist)
                f_difftab[i,0] = dist_tab[i]
#                print f_difftab[i]
            except:
                pass
        f_difftab = f_difftab[~np.isnan(f_difftab).any(axis=1)]
        print f_difftab
#        f_diff_interp = UnivariateSpline(f_difftab[:,0],f_difftab[:,1])
        try:
            f_diff_interp = interp1d(f_difftab[:,0], f_difftab[:,1], kind = 'cubic',
                                     bounds_error=False, fill_value=10000.)
        except:
            f_diff_interp = interp1d(f_difftab[:,0], f_difftab[:,1], kind = 'linear',
                                     bounds_error=False, fill_value=10000.)
                            
        bnds = [(f_difftab[0,0], f_difftab[-1,0])]
        dmax = minimize(f_diff_interp, 1., bounds=bnds)  
        
        return dmax.x
        

class Subhalo(object):        
    
    def J(self, dist, theta):
        """Theta in degrees and distance in kpc"""
        
        if theta > radtodeg * np.arctan(self.max_radius / dist):
            theta = np.arctan(self.max_radius / dist)
        else:
            theta = theta * np.pi / 180.
        
        if (self.los_max(0.,dist) - self.los_max(theta,dist)) > 10**-5:

            jfact = integrate.dblquad(lambda x,t: 2. * np.pi * kpctocm * np.sin(t) * 
                                      self.density(np.sqrt(dist**2. + x**2. - 2.0 * 
                                      dist * x * np.cos(t)))**2.0, 0., theta, 
                                      lambda x: self.los_min(x, dist), 
                                      lambda x: self.los_max(x, dist), epsabs=10**-5, 
                                      epsrel=10**-5) 

            return np.log10(jfact[0])
        else:
            return self.J_pointlike(dist)
                                                
        
           
    def J_pointlike(self, dist):
        
        jfact = integrate.quad(lambda x: 4. * np.pi * kpctocm  / dist**2. * self.density(x)**2. * x**2., 
                               0., self.max_radius, epsabs=10**-5, epsrel=10**-5)

        return np.log10(jfact[0])                   
      
    def los_min(self, theta, dist):
        return dist * np.cos(theta) - np.sqrt(dist**2. * (np.cos(theta)**2. - 1.) + self.max_radius**2.)
        
    def los_max(self, theta, dist):                                                        
        return dist * np.cos(theta) + np.sqrt(dist**2. * (np.cos(theta)**2. - 1.) + self.max_radius**2.)
 
    def Mass_in_R(self, r):    
        mass_enc = integrate.quad(lambda x: 4. * np.pi* x**2. * self.density(x) *
                                            GeVtoSolarM * (kpctocm)**3., 0., r)
        return mass_enc[0]
    
    def Mass_diff_005(self, rmax):       
        mass_enc = integrate.quad(lambda x: x**2. * self.density(x), 0., rmax)  
        return np.abs(4. * np.pi * GeVtoSolarM * (kpctocm)**3. * mass_enc[0] - 0.005 * self.halo_mass)

    def Truncated_radius(self):    
        bnds = [(10**-5, self.scale_radius)]
#       rtrunc = minimize(self.Mass_diff_005, [0.5 * self.scale_radius], bounds=bnds)
        rtrunc = brute(self.Mass_diff_005, bnds)
        return rtrunc
     
    def AngRad68(self, theta, dist):
        return np.abs(self.J(dist, theta) - self.J_pointlike(dist) - np.log10(0.68))
        
    def Spatial_Extension(self, dist):        
        bnds = [(10**-4., radtodeg * np.arctan(self.max_radius / dist))]
        extension = minimize(self.AngRad68, [0.9 * radtodeg * np.arctan(self.max_radius / dist)], args=(dist), method='SLSQP', bounds=bnds, tol=10**-2) 
        return extension.x
        
    def Full_Extension(self, dist):
        return radtodeg * np.arctan(self.max_radius / dist)

                
        

class Einasto(Subhalo):

    def __init__(self, halo_mass, alpha, concentration_param=None, 
                 z=0., truncate=False, arxiv_num=10070438):
        
        self.pname = 'Einasto_alpha_'+str(alpha)+'_C_params_'+str(arxiv_num) +\
                            '_Truncate_'+str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha
            
        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)
        
        self.c = concentration_param
            
        self.scale_radius = Virial_radius(self.halo_mass, M200=False) / self.c
        self.scale_density = ((self.halo_mass * self.alpha * np.exp(-2. / self.alpha) * 
                                (2. / self.alpha)**(3. / self.alpha)) / (4. * np.pi * 
                                self.scale_radius**3. * special.gamma(3. / self.alpha) *
                               (1. - special.gammaincc(3. / self.alpha, 2. * self.c**self.alpha / 
                               self.alpha))) * SolarMtoGeV * (cmtokpc)**3. )
                               
        if not truncate:
            self.max_radius = self.scale_radius
        else:
            self.max_radius = self.Truncated_radius()


    def density(self, r):        
        return self.scale_density * np.exp(-2. / self.alpha * (((r / self.scale_radius)**self.alpha) - 1.)) 
        

class NFW(Subhalo):

    def __init__(self, halo_mass, alpha, concentration_param=None, 
                 z=0., truncate=False, arxiv_num=10070438):
        """Note: alpha doesn't do anything, I'm keeping it here to be 
           consistent with Einasto profile
        """
        
        self.pname = 'NFW_alpha_'+'_C_params_'+str(arxiv_num) +\
                     '_Truncate_'+str(truncate)
        self.halo_mass = halo_mass
        self.alpha = alpha
            
        if concentration_param is None:
            concentration_param = Concentration_parameter(halo_mass, z, arxiv_num)
        
        self.c = concentration_param
            
        self.scale_radius = Virial_radius(self.halo_mass, M200=False) / self.c
        self.vir_rad = Virial_radius(self.halo_mass, M200=False)
        self.scale_density = ((self.halo_mass * SolarMtoGeV * (cmtokpc)**3.) /
                              (4. * np.pi * self.scale_radius ** 3. * 
                              (np.log(self.scale_radius + self.vir_rad) -
                              np.log(self.scale_radius) - self.vir_rad / 
                              (self.scale_radius + self.vir_rad))))
                               
        if not truncate:
            self.max_radius = self.scale_radius
        else:
            self.max_radius = self.Truncated_radius()


    def density(self, r):        
        return self.scale_density / ((r / self.scale_radius) * (1. + r / self.scale_radius)**2.)
        

class Observable(object):
    
    def __init__(self, mx, cross_sec, annih_prod, m_low=np.log10(6.48 * 10**6.), 
                 m_high=np.log10(2.0 * 10 **9), c_low=np.log10(20.),
                 c_high=2.1, alpha=0.16, profile=0, truncate=False, 
                 arxiv_num=10070438):
                  
        Profile_list = ["Einasto", "NFW"]
        self.mx = mx
        self.cross_sec = cross_sec
        self.annih_prod = annih_prod
        self.m_low = m_low
        self.m_high = m_high
        self.c_low = c_low
        self.c_high = c_high
        self.alpha = alpha
        self.profile = profile
        self.profile_name = Profile_list[self.profile]
        self.truncate = truncate
        self.arxiv_num = arxiv_num
        
        self.folder = MAIN_PATH + "/SubhaloDetection/Data/"
        info_str = "Observable_Profile_" + self.profile_name + "_Truncate_" +\
                    str(self.truncate) + "_mx_" + str(mx) + "_annih_prod_" +\
                    annih_prod + "_arxiv_num_" + str(arxiv_num) + "/"
        
        self.folder += info_str
        ensure_dir(self.folder)
                   
        default_dict = {'profile': 'Einasto', 'truncate': False, 'mx': 100., 'alpha': 0.16,
                        'annih_prod': 'BB', 'arxiv_num': 10070438, 'c_low': np.log10(20.),
                        'c_high': 2.1, 'c_num': 15, 'm_low': np.log10(6.48 * 10**6.), 
                        'm_high': np.log10(2.0 * 10 **9), 'm_num': 30}
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
                                c_num = 50):
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
                    
                    if not exists:
                        if subhalo.Full_Extension(d) > .05:                
                            extension_tab[ind] = subhalo.Spatial_Extension(d)
    
                extension_tab = extension_tab[np.nonzero(extension_tab)]
                entries_added = len(extension_tab)
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
    
    
    def Table_Dmax_Extended(self, m_num=20, c_num = 15):
        """ Tables d_max extended for future use. 
            
            ONLY NFW as of now
        """
        self.param_list['m_num'] = m_num
        self.param_list['c_num'] = c_num
        
        if os.path.isfile(self.folder+"param_list.pkl"):
            openfile = open(self.folder+"param_list.pkl", 'rb')
            old_dict = pickle.load(openfile)
            openfile.close()
            check_diff = DictDiffer(self.param_list, old_dict)
            if bool(check_diff.changed()):
                [os.remove(f) for f in os.listdir(self.folder)]
                pickle.dump(self.param_list, open(self.folder+"param_list.pkl", "wb")) 
        else:
            pickle.dump(self.param_list, open(self.folder+"param_list.pkl", "wb"))        
        
        Profile_names=['Einasto','NFW']
        
        file_name = 'Dmax_' + str(Profile_names[self.profile]) + '_Truncate_' +\
                    str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
                    str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' +\
                    str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'
                        
        mass_list = np.logspace(self.m_low, self.m_high, m_num)
        c_list = np.logspace(self.c_low, self.c_high, c_num)  
        
        for m in mass_list:
            print 'Subhalo mass: ', m
            for c in c_list:
                print 'Concentration parameter: ', c
                try:
                    look_array = np.loadtxt(self.folder + file_name)
                    if any((np.round([m,c],4) == x).all() for x in np.round(look_array[:,0:2],4)):
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
                                     profile=self.profile)
                    
                    dmax = dm_model.D_max_extend()           
                    tab = np.array([m, c, dmax[0]]).transpose()
                        
                    if os.path.isfile(self.folder+file_name):
                        load_info = np.loadtxt(self.folder + file_name)
                        add_to_table = np.vstack((load_info,tab))
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
            return (np.exp(-(np.log(c / cm) / (np.sqrt(2.0) * sigma_c)) ** 2.0) /
                    np.sqrt(2. * np.pi) * sigma_c * c)
           
        file_name = 'Dmax_' + str(Profile_names[self.profile]) + '_Truncate_' +\
                    str(self.truncate) + '_Cparam_' + str(self.arxiv_num) + '_alpha_' +\
                    str(self.alpha) + '_mx_' + str(self.mx) + '_cross_sec_' +\
                    str(np.log10(self.cross_sec)) + '_annih_prod_' + self.annih_prod + '.dat'
        
        
        integrand_table = np.loadtxt(self.folder+file_name)
        integrand_table[:,2] = (260. * (0.005 * integrand_table[:,0]) ** (-1.9) * 
                                prob_c(integrand_table[:,1],integrand_table[:,0]) * 
                                integrand_table[:,2]**3. / 3.0)
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
        integr = integrand.integral(3.24 * 10**4. / 0.005, 2.0 * 10**9., 10., 250.)
                                    
        # TODO: consider alternative ways of performing this integral
    
        return 4. * np.pi * (1. - np.sin(bmin * np.pi / 180.)) * integr
