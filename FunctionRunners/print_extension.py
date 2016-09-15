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
            jratio = 10. ** (halo.J(dist, 0.31) - halo.J_pointlike(dist))
            if jratio > 0.68:
                dtab = np.logspace(np.log10(dist) - 1, np.log10(.8 * dist), 3)
                jtab = np.zeros_like(dtab)
                for i, d in enumerate(dtab):
                    jtab = 10. ** (halo.J(d, 0.31) - halo.J_pointlike(d))
                dtab = np.append(dtab, dist)
                jtab = np.append(jtab, jratio)
                fit = np.polyfit(np.log10(dtab), np.log10(jtab), 1)
                dmax = 10 ** (- fit[1] / fit[0]) * 0.68 ** (1. / fit[0])
                sv_file[j] = [line[0], line[1], line[2], dmax]
            else:
                sv_file[j] = line
        np.savetxt(new_f, sv_file, fmt='%.3e')
