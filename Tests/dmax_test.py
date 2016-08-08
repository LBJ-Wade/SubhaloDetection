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
dist = 1.

a = Model(dm_mass, c_sec, annih_prod, subhalo_mass / 0.005, .16,
          truncate=True, arxiv_num=arxiv_num, profile=0,
          pointlike=False)

b = Model(dm_mass,c_sec, annih_prod, subhalo_mass / 0.005, .16,
          truncate=True, arxiv_num=arxiv_num, profile=1,
          pointlike=False)

c = Model(dm_mass, c_sec, annih_prod, subhalo_mass, .16,
          truncate=False, arxiv_num=arxiv_num, profile=0,
          pointlike=False)

d = Model(dm_mass, c_sec, annih_prod, subhalo_mass, .16,
          truncate=False, arxiv_num=arxiv_num, profile=1,
          pointlike=False)

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

print 'Profile: NFW \nTruncate'
print 'Total Flux: ', b.Total_Flux(dist)
print 'D_max Pointlike: ', b.d_max_point()
print 'Spectral Index: ', b.Determine_Gamma()
print 'Minimum Flux (Extended): ', b.min_Flux(dist)
print 'D_max Extended: ', b.D_max_extend()

print '\n'

print 'Profile: Einasto \nNot Truncated'
print 'Total Flux: ', c.Total_Flux(dist)
print 'D_max Pointlike: ', c.d_max_point()
print 'Spectral Index: ', c.Determine_Gamma()
print 'Minimum Flux (Extended): ', c.min_Flux(dist)
print 'D_max Extended: ', c.D_max_extend()


print '\n'

print 'Profile: NFW \nNot Truncated'
print 'Total Flux: ', d.Total_Flux(dist)
print 'D_max Pointlike: ', d.d_max_point()
print 'Spectral Index: ', d.Determine_Gamma()
print 'Minimum Flux (Extended): ', d.min_Flux(dist)
print 'D_max Extended: ', d.D_max_extend()


