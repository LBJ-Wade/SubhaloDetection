# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:44:48 2016

@author: SamWitte
"""

import numpy as np
import copy
from scipy import integrate
from scipy.interpolate import UnivariateSpline,interp1d
from subhalo import *

def Determine_Gamma(mx, annih_prod):
    """ Annih_prod can only be BB as of now"""
    num_collisions = 10**5.
    spec_file = MAIN_PATH + "/Spectrum/"+str(int(mx))+"GeVDMspectrum.dat"
    integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum"+annih_prod+".dat"
    
    spectrum = np.loadtxt(spec_file)
    spectrum[:,1] = spectrum[:,1] / num_collisions
     
    integrated_list = np.loadtxt(integrate_file)
    integrated_rate = interp1d(integrated_list[:,0],integrated_list[:,1])
    
    gamma_list = [1.5, 2.0, 2.5, 3.0]    
    normalize = np.zeros(len(gamma_list))
    for g in range(len(gamma_list)):
        normalize[g] = integrated_rate(mx) / integrate.quad(lambda x: x**(-gamma_list[g]), 1., mx)[0]

    meanE_s = UnivariateSpline(spectrum[:,0],spectrum[:,0] * spectrum[:,1])
    meanE = meanE_s.integral(1., mx)

    
    meanE_gam = np.zeros(len(gamma_list))
    for g in range(len(gamma_list)):
        meanE_gam[g] = integrate.quad(lambda x: normalize[g] * x**(-gamma_list[g] + 1.), 1., mx)[0]
    
    diff_meanE = np.abs(meanE_gam - meanE)

    return gamma_list[diff_meanE.argmin()]
    
    
def Threshold(gamma, extension):
    
    file = MAIN_PATH + "/ExtendedThresholds/DetectionThresholdGamma" +\
           str(int(gamma*10)) + ".dat"
            
    thresh_list = np.loadtxt(file)
    thresh_interp = UnivariateSpline(thresh_list[:,0],thresh_list[:,1])
    return thresh_interp(extension)
        
    
        
    