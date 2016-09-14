import os,sys
import glob
import numpy as np
sys.path.insert(0, os.environ['SUBHALO_MAIN_PATH']+'/SubhaloDetection')
from subhalo import *

dir = MAIN_PATH + '/SubhaloDetection/Data/Observable_Profile_HW_Extended_mx_' \
                  '15.0_annih_prod_BB_gamma_0.85_Conser__stiff_rb_False_Mlow_5.000e+00/*'

files = glob.glob(dir)

for f in files:
    my_data = np.loadtxt(f)
    my_data = np.vstack({tuple(row) for row in my_data})
    np.savetxt(f, my_data, fmt='%.3e')
    my_data = np.genfromtxt(f, dtype=None, names=["m", "r", "gam", "d"])
    ind = np.lexsort((my_data["gam"], my_data["r"], my_data["m"]))
    sorted_data = my_data[ind]
    sorted_data = np.asarray(sorted_data)
    np.savetxt(f, sorted_data, fmt='%.3e')
