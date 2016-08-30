# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:58:38 2016

@author: SamWitte
"""
import numpy as np
import os
from subhalo import *
from helper import *
from profiles import *


def table_spatial_extension(profile=0, truncate=False, arxiv_num=10070438,
                            M200=False, d_low=-3., d_high=1., d_num=30, m_num=20,
                            c_num=20, m_low=10.**4., m_high=10**7.):
    """ Tables spatial extension for future use.

        Profile Numbers correspond to [Einasto, NFW, HW] # 0 - 2
    """

    alpha = 0.16
    file_name = 'SpatialExtension_' + str(Profile_list[profile]) + '.dat'

    dir = MAIN_PATH + '/SubhaloDetection/Data/'

    if truncate:
        mass_list = np.logspace(np.log10(m_low / 0.005), np.log10(m_high / 0.005),
                                (np.log10(m_high) - np.log10(m_low)) * 6)
    else:
        mass_list = np.logspace(np.log10(m_low), np.log10(m_high),  (np.log10(m_high) - np.log10(m_low)) * 6)
    dist_list = np.logspace(d_low, d_high, d_num)
    if profile == 0:
        c_list = np.logspace(np.log10(2.5), 2.4, c_num)
    if profile == 2:
        rb_list = np.logspace(-3, np.log10(0.5), 20)
        gamma_list = np.linspace(0.2, 0.85 + 0.351 / 0.861 - 0.1, 20)

    for m in mass_list:
        print 'Subhalo mass: ', m
        if profile == 0:
            for c in c_list:
                print '     Concentration parameter: ', c
                subhalo = Einasto(m, alpha, c, truncate=True,
                                  arxiv_num=13131729, M200=False)
                for ind, d in enumerate(dist_list):
                    print '         Distance', d
                    value = '{:.4e}     {:.3e}      {:.4f}'.format(m, c, d)
                    with open(dir + file_name, 'a+') as f:
                        if not any(value == x.rstrip('\r\n') for x in f):
                            if subhalo.Full_Extension(d) > 0.1:
                                ext = subhalo.Spatial_Extension(d)
                                value += '      {:.4f} \n'.format(float(ext))
                                f.write(value)
        elif profile == 1:
            subhalo = NFW(m, 1., truncate=False,
                          arxiv_num=160106781, M200=True)
            for ind, d in enumerate(dist_list):
                print '         Distance', d
                value = '{:.4e}     {:.4f}'.format(m, d)
                with open(dir + file_name, 'a+') as f:
                    if not any(value == x.rstrip('\r\n') for x in f):
                        if subhalo.Full_Extension(d) > 0.1:
                            ext = subhalo.Spatial_Extension(d)
                            value += '      {:.4f} \n'.format(float(ext))
                            f.write(value)
        else:
            for rb in rb_list:
                print '     Rb: ', rb
                for gam in gamma_list:
                    print '         Gamma: ', gam
                    subhalo = HW_Fit(m, gam=gam, rb=rb, M200=True, gcd=8.5, stiff_rb=False)
                    for ind, d in enumerate(dist_list):
                        print '           Distance', d
                        value = '{:.4e}     {:.3e}      {:2f}      {:.4f}'.format(m, rb, gam, d)
                        f = np.loadtxt(dir + file_name)
                        m_check = float('{:.4e}'.format(m))
                        rb_check = float('{:.3e}'.format(rb))
                        gam_check = float('{:.2f}'.format(gam))
                        d_check = float('{:.4f}'.format(d))
                        if np.sum((f[:, 0] == m_check) & (f[:, 1] == rb_check) &
                                          (f[:, 2] == gam_check) & (f[:, 3] == d_check)) < 1:
                            if subhalo.Full_Extension(d) > 0.1:
                                ext = subhalo.Spatial_Extension(d)
                                print '             Extension: ', ext
                                value += '      {:.4f} \n'.format(float(ext))
                                f.write(value)

    return
