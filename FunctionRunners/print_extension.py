import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from subhalo import *
import numpy as np
import glob

dir = os.environ['SUBHALO_MAIN_PATH'] + \
      '/SubhaloDetection/Data/Observable_Profile_HW_Extended_' \
      'mx_15.0_annih_prod_BB_gamma_0.85_Conser__stiff_rb_False_Mlow_5.000e+00/*'

ext_lim = 0.31

files = glob.glob(dir)
for f in files:
    print f
    load = np.loadtxt(f)
    new_f = f[:-4] + 'Extension_Lim_{:.2f}.dat'.format(ext_lim)
    sv_file = np.zeros_like(load)
    if os.path.isfile(new_f):
        pass
    else:
        for j, line in enumerate(load):
            print j + 1, '/', len(load)
            halo = HW_Fit(line[0], gam=line[2], rb=line[1])
            dist = line[3]
            ext = halo.Spatial_Extension(dist)
            if ext < 0.31:
                test_dist = np.log10(dist) - 2.
                ext_test = halo.Spatial_Extension(10. ** test_dist)
                if ext_test == 2.0:
                    min_dist = np.log10(dist) - 1.5
                    max_dist = np.log10(dist * 0.8)
                elif ext_test == 0.1:
                    min_dist = np.log10(dist) - 4.
                    max_dist = np.log10(dist) - 2.
                else:
                    min_dist = np.log10(dist) - 2.2
                    max_dist = np.log10(dist) - 0.5
                dist_tab = np.logspace(min_dist, max_dist, 7)
                se_tab = np.zeros_like(dist_tab)
                for i,d in enumerate(dist_tab):
                    se_tab[i] = halo.Spatial_Extension(d)
                dist_tab = dist_tab[(se_tab > 0.1) & (se_tab < 2.)]
                se_tab = se_tab[(se_tab > 0.1) & (se_tab < 2.)]
                if ext > 0.1:
                    dist_tab = np.append(dist_tab, dist)
                    se_tab = np.append(se_tab, ext)
                fit = np.polyfit(np.log10(dist_tab), np.log10(se_tab), 1)
                dmax = 10. ** (- fit[1] / fit[0]) * ext_lim ** (1. / fit[0])
                sv_file[j] = [line[0], line[1], line[2], dmax]
            else:
                sv_file[j] = line
        np.savetxt(new_f, sv_file, fmt='%.3e')
