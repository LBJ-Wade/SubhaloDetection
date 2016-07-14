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
from scipy.optimize import minimize,minimize_scalar


class Model(object):
    
    def __init__(self, mx, cross_sec, annih_prod):
        self.mx = mx
        self.c_sec = cross_sec
        self.annih_prod = annih_prod


class Subhalo(object):        
    
    def J(self, dist, theta):
        """Theta in degrees and distance in kpc"""
        
        if theta > radtodeg * np.arctan(self.max_radius / dist):
            theta = np.arctan(self.max_radius / dist)
        else:
            theta = theta * np.pi / 180.
        
        if (self.los_max(0.,dist) - self.los_max(theta,dist)) > 10**-6:
            jfact = integrate.dblquad(lambda x,t: 2. * np.pi * kpctocm * np.sin(t) * 
                                      self.density(np.sqrt(dist**2. + x**2. - 2.0 * 
                                      dist * x * np.cos(t)))**2.0, 0., theta, 
                                      lambda x: self.los_min(x, dist), 
                                      lambda x: self.los_max(x, dist), epsabs=10**-4, 
                                      epsrel=10**-4)  
        else:
            jfact = [10**self.J_pointlike(dist)]
                                                
        return np.log10(jfact[0])
           
    def J_pointlike(self, dist):
        
        jfact = integrate.quad(lambda x: 4. * kpctocm * np.pi / dist**2. * self.density(x)**2. * x**2., 
                               0., self.max_radius, epsabs=10**-4, epsrel=10**-4)

        return np.log10(jfact[0])                   
      
    def los_min(self, theta, dist):
        return dist * np.cos(theta) - np.sqrt(dist**2. * (np.cos(theta)**2. - 1.) + self.max_radius**2.)
        
    def los_max(self, theta, dist):                                                        
        return dist * np.cos(theta) + np.sqrt(dist**2. * (np.cos(theta)**2. - 1.) + self.max_radius**2.)
 
    def Mass_diff_005(self, rmax):
        
        mass_enc = integrate.quad(lambda x: x**2. * self.density(x), 0., rmax)
        
        return np.abs(4. * np.pi * GeVtoSolarM * (kpctocm)**3. * mass_enc[0] - 0.005 * self.halo_mass)

    def Truncated_radius(self): 
        
        bnds = [(10**-5, self.scale_radius)]
        rtrunc = minimize(self.Mass_diff_005, [0.5 * self.scale_radius], bounds=bnds)
        return rtrunc.x
     
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
            
        self.scale_radius = Virial_radius(self.halo_mass) / self.c
        self.scale_density = ((self.halo_mass * self.alpha * np.exp(-2. / self.alpha) * 
                                (2. / self.alpha)**(3. / self.alpha)) / (4. * np.pi * 
                                self.scale_radius**3. * special.gamma(3. / self.alpha) *
                               (1. - special.gammaincc(3. / self.alpha, 2. * self.c**self.alpha / 
                               self.alpha))) * SolarMtoGeV * (cmtokpc)**3. )
                               
        if not truncate:
            self.max_radius = Virial_radius(self.halo_mass)
        else:
            self.max_radius = self.Truncated_radius()


    def density(self, r):        
        return self.scale_density * np.exp(-2. / self.alpha * (((r / self.scale_radius)**self.alpha) - 1.))      
      
    
def Table_Spatial_Extension(d_low=-3., d_high=1., d_num=80, m_low=np.log10(6.48 * 10**6.), 
                            m_high=np.log10(2.0 * 10 **9), m_num=60, c_low=np.log10(20.),
                            c_high=2.4, c_num = 50, alpha=0.16, profile=0, 
                            truncate=False, arxiv_num=10070438):
    """ Tables spatial extension for future use. 
        
        Profile Numbers correspond to [Einasto, NFW] # 0 - 1
    """
    Profile_names=['Einasto','NFW']
    
    folder = "/Users/SamWitte/Desktop/SubhaloProject/SubhaloDetection/Data/"
    file_name = str(Profile_names[profile]) + '_Truncate_' + str(truncate) +\
                '_Cparam_' + str(arxiv_num) + '_alpha_' + str(alpha) + '.dat'
                    
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