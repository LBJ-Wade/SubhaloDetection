import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from subhalo import *
from profiles import Einasto, NFW, HW_Fit
import scipy.integrate as integrate
import numpy as np
import time

dist = 1.
mass = 1.*10**7.
a = Einasto(mass / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)
b = NFW(mass / 0.005, .16, truncate=True, arxiv_num=13131729, M200=False)

cc = Einasto(mass, .16, truncate=False, arxiv_num=160106781, M200=True)
dd = NFW(mass, .16, truncate=False, arxiv_num=160106781, M200=True)

HW = HW_Fit(mass)
HW_gu = HW_Fit(mass, gam=1.25)
HW_gd = HW_Fit(mass, gam=0.426)

print 'Mass: ', mass
print 'Dist: ', dist

print '\n arXiv Num: 13131729 \n'

print 'Einasto Truncate:\n'
print '     Scale Radius: ', a.scale_radius
print '     Max Radius: ', a.max_radius
print '     Mass in max radius: ', a.Mass_in_R(a.max_radius)
print '     J pointlike: ', 10. ** a.J_pointlike(dist)
ext = a.Spatial_Extension(dist)
print '     Spatial Extension: ', ext
print '     J_ext(se)/J_point: ', 10. ** a.J(dist, ext) / 10. ** a.J_pointlike(dist)
print '     Full Extension: ', a.Full_Extension(dist)





print '\n arXiv Num: 160106781 \n',


# print '     Einasto Non-Truncated: \n'
# print '     Scale Radius: ', cc.scale_radius
# print '     Max Radius: ', cc.max_radius
# print '     Mass in max radius: ', cc.Mass_in_R(c.max_radius)
# print '     J pointlike: ', 10. ** cc.J_pointlike(dist)
# ext = cc.Spatial_Extension(dist)
# print '     Spatial Extension: ', ext
# print '     J_ext(se)/J_point: ', 10. ** cc.J(dist, ext) / 10. ** cc.J_pointlike(dist)
# print '     Full Extension: ', cc.Full_Extension(dist)
#
# print '\n'

print 'NFW Non-Truncated:\n'
print '     Scale Radius: ', dd.scale_radius
print '     Max Radius: ', dd.max_radius
print '     Mass in max radius: ', dd.Mass_in_R(dd.max_radius)
print '     J pointlike: ',10. ** dd.J_pointlike(dist)
ext = dd.Spatial_Extension(dist)
print '     Spatial Extension: ', ext
print '     J_ext(se)/J_point: ', 10. ** dd.J(dist, ext) / 10. ** dd.J_pointlike(dist)
print '     Full Extension: ', dd.Full_Extension(dist)

print '\n'

print 'HW Fit: \n'
print '     Gamma, Rb: ', HW.gam, HW.rb
print '     Max Radius: ', HW.max_radius
print '     Mass in max radius: ', HW.Mass_in_R(HW.max_radius)
print '     J pointlike: ',10. ** HW.J_pointlike(dist)
ext = HW.Spatial_Extension(dist)
print '     Spatial Extension: ', ext
print '     J_ext(se)/J_point: ', 10. ** HW.J(dist, ext) / 10. ** HW.J_pointlike(dist)
print '     Full Extension: ', HW.Full_Extension(dist)

print '\n'


# print 'HW Fit High Gamma: \n'
# print '     Gamma, Rb: ', HW_gu.gam, HW_gu.rb
# print '     Max Radius: ', HW_gu.max_radius
# print '     Mass in max radius: ', HW_gu.Mass_in_R(HW2.max_radius)
# print '     J pointlike: ',10. ** HW_gu.J_pointlike(dist)
# ext = HW_gu.Spatial_Extension(dist)
# print '     Spatial Extension: ', ext
# print '     J_ext(se)/J_point: ', 10. ** HW_gu.J(dist, ext) / 10. ** HW_gu.J_pointlike(dist)
# print '     Full Extension: ', HW_gu.Full_Extension(dist)

print '\n'

print 'HW Fit Low Gamma: \n'
print '     Gamma, Rb: ', HW_gd.gam, HW_gd.rb
print '     Max Radius: ', HW_gd.max_radius
print '     Mass in max radius: ', HW_gd.Mass_in_R(HW_gd.max_radius)
print '     J pointlike: ',10. ** HW_gd.J_pointlike(dist)
ext = HW_gd.Spatial_Extension(dist)
print '     Spatial Extension: ', ext
print '     J_ext(se)/J_point: ', 10. ** HW_gd.J(dist, ext) / 10. ** HW_gd.J_pointlike(dist)
print '     Full Extension: ', HW_gd.Full_Extension(dist)


#
# vmax = 5.
# gcd = 20.
# print 'Vmax, GC Dist: ', vmax, gcd
#
# ddd = NFW(mass, .16, truncate=False, arxiv_num=160304057, M200=True, vmax=vmax, gcd=gcd)
# print '     Scale Radius: ', ddd.scale_radius
# print '     Max Radius: ', ddd.max_radius
# print '     Mass in max radius: ', ddd.Mass_in_R(a.max_radius)
# print '     J pointlike: ', 10. ** ddd.J_pointlike(dist)
# ext = ddd.Spatial_Extension(dist)
# print '     Spatial Extension: ', ext
# print '     J_ext(se)/J_point: ', 10. ** ddd.J(dist, ext) / 10. ** ddd.J_pointlike(dist)
# print '     Full Extension: ', ddd.Full_Extension(dist)
