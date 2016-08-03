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
    
def Virial_radius(Mass):
    return 200. * (Mass / (2. * 10 ** 12.)) ** (1. / 3.)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class DictDiffer(object):
    """
        Calculate the difference between two dictionaries as:
        (1) items added
        (2) items removed
        (3) keys same in both but changed values
        (4) keys same in both and unchanged values
        """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)
    def added(self):
        return self.set_current - self.intersect
    def removed(self):
        return self.set_past - self.intersect
    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])
    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])