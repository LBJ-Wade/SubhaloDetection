# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

@author: SamWitte
"""
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.integrate import quad
import glob



try:
    MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'

#Conversions
SolarMtoGeV = 1.11547 * 10**(57.)
GeVtoSolarM = 8.96485 * 10**(-58.)
cmtokpc = 3.24078 * 10**(-22.)
kpctocm = 3.08568 * 10**(21.)
degtorad = np.pi / 180.
radtodeg = 180. / np.pi

#Numerical Quantities -- taken from PDG
hubble = 0.673
rho_critical = 2.775 * 10**(11.) * hubble**2. * 10.**-9. #  Units: Solar Mass / (kpc)^3
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


def interpola(val, x, y):
    try:
        f = np.zeros(len(val))
        for i, v in enumerate(val):
            if v <= x[0]:
                f[i] = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0])
            elif v >= x[-1]:
                f[i] = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-2])
            else:
                f[i] = interp1d(x, y, kind='cubic').__call__(v)
    except TypeError:
        if val <= x[0]:
            f = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (val - x[0])
        elif val >= x[-1]:
            f = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (val - x[-2])
        else:
            try:
                f = interp1d(x, y, kind='cubic').__call__(val)
            except:
                f = interp1d(x, y, kind='linear').__call__(val)
    return f


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return


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


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)
    return


def str2bool(v):
    if type(v) == bool:
        return v
    elif type(v) == str:
        return v.lower() in ("yes", "true", "t", "1")


def table_gamma_index(annih_prod='BB'):
    #  TODO:
    #  Use glob to determine what spectrum files exist...
    #  Also make spectrum for huge range of mass
    mass_tab = np.array([34., 40., 100.])#np.logspace(1., 3., 100)
    num_collisions = 10 ** 5.
    gamma_mx = np.zeros(mass_tab.size)
    integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + annih_prod + ".dat"

    for i,mx in enumerate(mass_tab):
        spec_file = MAIN_PATH + "/Spectrum/" + str(int(mx)) + "GeVDMspectrum.dat"
        spectrum = np.loadtxt(spec_file)
        spectrum[:, 1] = spectrum[:, 1] / num_collisions

        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1])

        gamma_list = np.linspace(1.5, 3.0, 100)
        normalize = np.zeros(len(gamma_list))
        for g in range(len(gamma_list)):
            normalize[g] = integrated_rate(mx) / quad(lambda x: x ** (-gamma_list[g]), 1., mx)[0]

        def int_meanes(x):
            return interpola(x, spectrum[:, 0], spectrum[:, 0] * spectrum[:, 1])
        meanE = quad(int_meanes, 1., mx, epsabs=10.**-4., epsrel=10.**-4.)[0]

        meanE_gam = np.zeros(len(gamma_list))
        for g in range(len(gamma_list)):
            meanE_gam[g] = quad(lambda x: normalize[g] * x ** (-gamma_list[g] + 1.), 1., mx,
                                          epsabs=10.**-4., epsrel=10.**-4.)[0]

        diff_meanE = np.abs(meanE_gam - meanE)
        gamma_mx[i] = gamma_list[diff_meanE.argmin()]

    sv_dir = MAIN_PATH + "/SubhaloDetection/Data/Misc_Items/"
    file_name = 'GammaIndex_given_mx_for_annih_prod_' + annih_prod + '.pdf'
    np.savetxt(sv_dir + file_name, [mass_tab, gamma_mx])
    return
