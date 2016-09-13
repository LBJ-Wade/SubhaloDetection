import os,sys
from subhalo import *
import glob
import numpy as np

dir = MAIN_PATH + '/SubhaloDetection/Data/Observable_Profile_HW_Extended_mx_' \
                  '15.0_annih_prod_BB_gamma_0.85_Conser__stiff_rb_False_Mlow_5.000e+00/*'

files = glob.glob(dir)

for f in files:
    my_data = np.genfromtxt(f, dtype=None, names=["m", "r", "gam", "d"])
    ind = np.lexsort((my_data["gam"], my_data["r"], my_data["m"]))
    sorted_data = my_data[ind]
    sorted_data = np.asarray(sorted_data)
    np.savetxt(f, sorted_data, fmt='%.3e')
