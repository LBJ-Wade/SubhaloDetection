# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:30:46 2016

@author: SamWitte
"""

import os, sys
sys.path.insert(0 , os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection')
from subhalo import *
import subhalo
import numpy as np
import time

annih_prod = 'BB'
dm_mass = 100.
c_sec = 3. * 10**-26.
subhalo_mass = 1. * 10. ** 7.
arxiv_num = 13131729
dist = 10.

a = Model(dm_mass, c_sec, annih_prod, subhalo_mass / 0.005, .16,
          truncate=True, arxiv_num=13131729, profile=0,
          pointlike=False)


b = Model(dm_mass, c_sec, annih_prod, subhalo_mass, .16,
          truncate=False, arxiv_num=160106781, profile=1,
          pointlike=False)

hw = Model(dm_mass, c_sec, annih_prod, subhalo_mass, profile=2,
           gam=0.945)


hw_gh = Model(dm_mass, c_sec, annih_prod, subhalo_mass, profile=2,
           gam=1.316)

hw_gl = Model(dm_mass, c_sec, annih_prod, subhalo_mass, profile=2,
           gam=0.426)

print 'Parameters: '
print 'Dark Matter Mass: ', dm_mass
print 'Annihilation Products: ', annih_prod
print 'Cross Section: ', c_sec
print 'Post TS Subhalo Mass: ', subhalo_mass
print 'Concentration Parameter Reference: ', arxiv_num
print 'Distance: ', dist

print '\n'

print 'Profile: Einasto \nTruncate'
print 'Total Flux: ', a.Total_Flux(dist)
print 'D_max Pointlike: ', a.d_max_point()
print 'Spectral Index: ', a.Determine_Gamma()
print 'Minimum Flux (Extended): ', a.min_Flux(dist)
print 'D_max Extended: ', a.D_max_extend()


print '\n'

print 'Profile: NFW \nNot Truncate'
print 'Total Flux: ', b.Total_Flux(dist)
print 'D_max Pointlike: ', b.d_max_point()
print 'Spectral Index: ', b.Determine_Gamma()
print 'Minimum Flux (Extended): ', b.min_Flux(dist)
print 'D_max Extended: ', b.D_max_extend()

print '\n'

print 'Profile: HW Standard \n'
print 'Total Flux: ', hw.Total_Flux(dist)
print 'D_max Pointlike: ', hw.d_max_point()
print 'Spectral Index: ', hw.Determine_Gamma()
print 'Minimum Flux (Extended): ', hw.min_Flux(dist)
print 'D_max Extended: ', hw.D_max_extend()



print '\n'

print 'Profile: HW \n Gamma = 1.316'
print 'Total Flux: ', hw_gh.Total_Flux(dist)
print 'D_max Pointlike: ', hw_gh.d_max_point()
print 'Spectral Index: ', hw_gh.Determine_Gamma()
print 'Minimum Flux (Extended):', hw_gh.min_Flux(dist)
print 'D_max Extended: IGNORE'#, hw_c.D_max_extend()


print '\n'

print 'Profile: HW \n Gamma = 0.426'
print 'Total Flux: ', hw_gl.Total_Flux(dist)
print 'D_max Pointlike: ', hw_gl.d_max_point()
print 'Spectral Index: ', hw_gl.Determine_Gamma()
print 'Minimum Flux (Extended): ', hw_gl.min_Flux(dist)
print 'D_max Extended: ', hw_gl.D_max_extend()


