# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 09:58:38 2016

@author: SamWitte
"""
import numpy as np
import os
from subhalo import *
from helper import *
from profiles import NFW, Einasto


def table_spatial_extension(profile=0, truncate=False, arxiv_num=10070438,
                            M200=False, d_low=-3., d_high=1., d_num=30, m_num=20,
                            c_num=20):
    """ Tables spatial extension for future use.

        Profile Numbers correspond to [Einasto, NFW] # 0 - 1
    """
    Profile_names = ['Einasto', 'NFW']
    alpha = 0.16
    file_name = 'SpatialExtension_' + str(Profile_names[profile]) + '_Truncate_' + \
                str(truncate) + '_Cparam_' + str(arxiv_num) + '_alpha_' + \
                str(alpha) + '.dat'
    dir = MAIN_PATH + '/SubhaloDetection/Data/'

    if truncate:
        mass_list = np.logspace(np.log10(3.24 * 10. ** 4. / 0.005),
                                np.log10(2. * 10. ** 9.), m_num)
    else:
        mass_list = np.logspace(np.log10(3.24 * 10. ** 4.), np.log10(1. * 10. ** 7.), m_num)
    dist_list = np.logspace(d_low, d_high, d_num)
    c_list = np.logspace(np.log10(2.5), 2.4, c_num)
    for m in mass_list:
        print 'Subhalo mass: ', m
        for c in c_list:
            print '     Concentration parameter: ', c

            extension_tab = np.zeros(len(dist_list))
            if profile == 0:
                subhalo = Einasto(m, alpha, c, truncate=truncate,
                                  arxiv_num=arxiv_num, M200=M200)
            elif profile == 1:
                subhalo = NFW(m, alpha, c, truncate=truncate,
                              arxiv_num=arxiv_num, M200=M200)
            else:
                'Profile not implemented yet'

            for ind, d in enumerate(dist_list):
                print '         Distance', d
                try:
                    look_array = np.loadtxt(dir + file_name)
                    if any((np.round([m, c, d], 5) == x).all() for x in np.round(look_array[:, 0:3], 5)):
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

            if os.path.isfile(dir + file_name):
                load_info = np.loadtxt(dir + file_name)
                add_to_table = np.vstack((load_info, full_tab))
                np.savetxt(dir + file_name, add_to_table)
            else:
                np.savetxt(dir + file_name, full_tab)
    return
