import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from subhalo import *
from profile import *
from helper import *
import scipy.integrate as integrate
import numpy as np
import time

dist = 10.
mass = 1.*10**7.
a = Einasto(mass / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
b = NFW(mass / 0.005, .16, truncate=True, arxiv_num=13131729)
c = Einasto(mass, .16, truncate=False)
d = NFW(mass, .16, truncate=False, arxiv_num=13131729, M200=False)

print 'Mass: ', mass
print 'Dist: ',dist

print '\n'

print 'Einasto Truncate:\n'
print 'Mass in R_vir: ', a.Mass_in_R(Virial_radius(a.halo_mass))
print 'Scale Radius: ', a.scale_radius
print 'Max Radius: ', a.max_radius
print 'Mass in max radius: ', a.Mass_in_R(a.max_radius)
print 'J pointlike: ', 10. ** a.J_pointlike(dist)
print 'Spatial Extension: ', a.Spatial_Extension(dist)

print '\n'

print 'NFW Truncate:\n'
print 'Mass in R_vir: ', b.Mass_in_R(Virial_radius(b.halo_mass))
print 'Scale Radius: ', b.scale_radius
print 'Max Radius: ', b.max_radius
print 'Mass in max radius: ', b.Mass_in_R(b.max_radius)
print 'J pointlike: ',10. ** b.J_pointlike(dist)
print 'Spatial Extension: ', b.Spatial_Extension(dist)

print '\n'

print 'Einasto Non-Truncated: \n'
print 'Mass in R_vir: ', c.Mass_in_R(Virial_radius(c.halo_mass))
print 'Scale Radius: ', c.scale_radius
print 'Max Radius: ', c.max_radius
print 'Mass in max radius: ', c.Mass_in_R(c.max_radius)
print 'J pointlike: ', 10. ** c.J_pointlike(dist)
print 'Spatial Extension: ', c.Spatial_Extension(dist)

print '\n'

print 'NFW Non-Truncated:\n'
print 'Mass in R_vir: ', d.Mass_in_R(Virial_radius(d.halo_mass))
print 'Scale Radius: ', d.scale_radius
print 'Max Radius: ', d.max_radius
print 'Mass in max radius: ', d.Mass_in_R(d.max_radius)
print 'J pointlike: ',10. ** d.J_pointlike(dist)
print 'Spatial Extension: ', d.Spatial_Extension(dist)

