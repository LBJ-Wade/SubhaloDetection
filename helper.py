# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

@author: SamWitte
"""
import numpy as np
import os

#Conversions
SolarMtoGeV = 1.11547 * 10**(57.)
GeVtoSolarM = 8.96485 * 10**(-58.)
cmtokpc = 3.24078 * 10**(-22.)
kpctocm = 3.08568 * 10**(21.)
degtorad = np.pi / 180.
radtodeg = 180. / np.pi

#Numerical Quantities
hubble = 0.673
rho_critical = 2.775 * 10**(11.) * hubble**2.
delta_200 = 200.


def Concentration_parameter(Mass, z=0, arxiv_num=13131729):
    """ Mass input in Solar Masses. Two different possible relations below."""

    c=0.    
    if arxiv_num == 13131729:
        if z!= 0:
            raise ValueError 
        coeff_array = np.array([37.5153, -1.5093, 1.636 * 10**(-2),
                                3.66 * 10**(-4), -2.89237 * 10**(-5), 
                                5.32 * 10**(-7)])
        for x in range(coeff_array.size):
            c += coeff_array[x] * (np.log(Mass * hubble))**x
    elif arxiv_num == 10070438:
        w = 0.029
        m = 0.097
        alpha = -110.001
        beta = 2469.720
        gamma = 16.885
        
        a = w * z - m
        b = alpha / (z + gamma) + beta / (z + gamma)**2.
        
        c = (Mass * hubble)**a * 10.**b
    
    else:
        print 'Wrong arXiv Number for Concentration Parameter'
    
    return c
    
def Virial_radius(Mass, M200=False):
    """If M200 is None, virial radius scaled version of Milky Way. Otherwise,
       taken to be M200"""
        
    if not M200:
        return 200. * (Mass / (2. * 10**(12.)))**(1./3.)
    else:
        return ((3. * Mass) / (4. * np.pi * delta_200 * 
                rho_critical / 10**9))**(1./3.)


