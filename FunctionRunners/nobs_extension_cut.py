import os,sys
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from subhalo import *
import numpy as np
import glob

path =os.environ['SUBHALO_MAIN_PATH'] + '/SubhaloDetection/'
profile = 2
annih_prod = 'BB'
mx = 15
ext_lim = 0.31
bmin = 20.
extra_tag = '_Gamma_0.850_Stiff_rb_False'
simga_n_file = 'HW' + '_mx_' + str(mx) + '_annih_prod_' + annih_prod + '_bmin_' +\
               str(bmin) + '_Extended' + extra_tag + '_Mlow_{:.3f}'.format(10**5.) +\
               '_Ext_Lim_{:.2f}'.format(ext_lim) + '.dat'


dir = os.environ['SUBHALO_MAIN_PATH'] + \
      '/SubhaloDetection/Data/Observable_Profile_HW_Extended_' \
      'mx_{:.1f}_annih_prod_BB_gamma_0.85_Conser__stiff_rb_False_'.format(mx) +\
      'Mlow_5.000e+00/*Extension_Lim_{:.2f}*'.format(ext_lim)

files = glob.glob(dir)

for f in files:
    cs_label = 'cross_sec_'
    loc = f.find(cs_label)
    cross_sec = float(f[loc + len(cs_label): loc + len(cs_label) + 9])
    build_obs = Observable(mx, cross_sec, annih_prod, m_low=5.,
                           m_high=7., profile=2, truncate=False,
                           point_like=False, m200=True)

    n_ext_obs = build_obs.N_obs(bmin, plike=False)
    if os.path.isfile(path + '/Data/Cross_v_Nobs/' + simga_n_file):
        cross_sec_nobs = np.loadtxt(path + '/Data/Cross_v_Nobs/' + simga_n_file)
        add_to_table = np.vstack((cross_sec_nobs, [cross_sec, n_ext_obs]))
        save_tab = add_to_table[np.lexsort(np.fliplr(add_to_table).T)]
        np.savetxt(path + '/Data/Cross_v_Nobs/' + simga_n_file, save_tab)
    else:
        np.savetxt(path + '/Data/Cross_v_Nobs/' + simga_n_file, np.array([cross_sec, n_ext_obs]))
