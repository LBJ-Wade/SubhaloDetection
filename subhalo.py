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
from scipy.optimize import minimize,minimize_scalar,brute
import os

MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']

class Model(object):
    """NOT READY"""
    def __init__(self, mx, cross_sec, annih_prod, halo_mass, alpha, 
                 concentration_param=None, z=0., truncate=False, 
                 arxiv_num=10070438):
        self.mx = mx
        self.c_sec = cross_sec
        self.annih_prod = annih_prod
        
        self.subhalo = Einasto(halo_mass, alpha, 
                               concentration_param=concentration_param, 
                               truncate=truncate, arxiv_num=arxiv_num)

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

        dist_tab = np.logspace(-2., 1., 80)
        
        f_difftab = np.zeros((dist_tab.size,2))

        for i,dist in enumerate(dist_tab):
            try:
                f_difftab[i,1] = flux_diff(dist)
                f_difftab[i,0] = dist_tab[i]
                print f_difftab[i]
            except:
                pass
        f_difftab = f_difftab[~np.isnan(f_difftab).any(axis=1)]
        f_diff_interp = UnivariateSpline(f_difftab[:,0],f_difftab[:,1])
        bnds = [(10**f_difftab[0,0], 10**f_difftab[-1,0])]
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
        
        self.profile_name = 'Einasto_alpha_'+str(alpha)+'_C_params_'+str(arxiv_num) +\
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
            self.max_radius = Virial_radius(self.halo_mass, M200=False)
        else:
            self.max_radius = self.Truncated_radius()


    def density(self, r):        
        return self.scale_density * np.exp(-2. / self.alpha * (((r / self.scale_radius)**self.alpha) - 1.)) 
        
    
def Table_Spatial_Extension(d_low=-3., d_high=1., d_num=80, m_low=np.log10(6.48 * 10**6.), 
                            m_high=np.log10(2.0 * 10 **9), m_num=60, c_low=np.log10(20.),
                            c_high=2.4, c_num = 50, alpha=0.16, profile=0, 
                            truncate=False, arxiv_num=10070438):
    """ Tables spatial extension for future use. 
        
        Profile Numbers correspond to [Einasto, NFW] # 0 - 1 (TODO: add NFW)
    """
    Profile_names=['Einasto','NFW']
    
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    file_name = 'SpatialExtension_' + str(Profile_names[profile]) + '_Truncate_' +\
                str(truncate) + '_Cparam_' + str(arxiv_num) + '_alpha_' +\
                str(alpha) + '.dat'
                    
    mass_list = np.logspace(m_low, m_high, m_num)
    dist_list = np.logspace(d_low, d_high, d_num)
    c_list = np.logspace(c_low, c_high, c_num)  
    
    for m in mass_list:
        print 'Subhalo mass: ', m
        for c in c_list:
            print 'Concentration parameter: ', c
            
            extension_tab = np.zeros(len(dist_list))
            if profile == 0:
                subhalo = Einasto(m, alpha, c, truncate=truncate, arxiv_num=arxiv_num)
            else:
                'Profile not implimented yet'
            
            for ind,d in enumerate(dist_list):
                try:
                    look_array = np.loadtxt(folder + file_name)
                    if any(([m,c,d] == x).all() for x in look_array[:,0:3]):
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
                
            if os.path.isfile(folder+file_name):
                load_info = np.loadtxt(folder + file_name)
                add_to_table = np.vstack((load_info,full_tab))
                np.savetxt(folder + file_name, add_to_table)
            else:
                np.savetxt(folder + file_name, full_tab)
    return


def Table_Dmax_Extended(mx, cross_sec, annih_prod, m_low=np.log10(6.48 * 10**6.),
                        m_high=np.log10(2.0 * 10 **9), m_num=60, 
                        c_low=np.log10(20.), c_high=2.4, c_num = 50, 
                        alpha=0.16, truncate=False, arxiv_num=10070438):
    """ Tables d_max extended for future use. 
        
        ONLY NFW as of now
    """
    Profile_names=['Einasto','NFW']
    
    folder = MAIN_PATH + "/SubhaloDetection/Data/"
    file_name = 'Dmax_' + str(Profile_names[0]) + '_Truncate_' +\
                str(truncate) + '_Cparam_' + str(arxiv_num) + '_alpha_' +\
                str(alpha) + '.dat'
                    
    mass_list = np.logspace(m_low, m_high, m_num)
    c_list = np.logspace(c_low, c_high, c_num)  
    
    for m in mass_list:
        print 'Subhalo mass: ', m
        for c in c_list:
            print 'Concentration parameter: ', c
            try:
                look_array = np.loadtxt(folder + file_name)
                if any(([m,c] == x).all() for x in look_array[:,0:2]):
                    exists = True
                else:
                    exists = False
            except:
                exists = False
                
            if not exists:  
                dm_model = Model(mx, cross_sec, annih_prod, m, alpha, 
                                 concentration_param=c, truncate=truncate, 
                                 arxiv_num=arxiv_num)
                
                dmax = dm_model.D_max_extend()           
                tab = np.array([m, c, dmax[0]]).transpose()
                    
                if os.path.isfile(folder+file_name):
                    load_info = np.loadtxt(folder + file_name)
                    add_to_table = np.vstack((load_info,tab))
                    np.savetxt(folder + file_name, add_to_table)
                else:
                    np.savetxt(folder + file_name, tab)        
    return
