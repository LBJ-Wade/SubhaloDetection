# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:54:38 2016

@author: SamWitte
"""
import numpy as np
import os
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import quad
from scipy.special import hyp2f1, gammaincc, gamma
import glob
import re


try:
    MAIN_PATH = os.environ['SUBHALO_MAIN_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'

via_lactea = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ViaLacteaII_Info.dat')
#via_lactea = np.loadtxt(MAIN_PATH + '/SubhaloDetection/Data/Misc_Items/ViaLacteaII_Useable_Subhalos.dat')
#def vmax_interp(x):
#    return interpola(x, np.log10(via_lactea[:, 5]), np.log10(via_lactea[:, 3]))
#def rmax_interp(x):
#    return interpola(x, np.log10(via_lactea[:, 5]), np.log10(via_lactea[:, 4]))
def tidal_interp(x):
    return interp1d(np.log10(via_lactea[:, 5]), np.log10(via_lactea[:, 6]),
                    bounds_error=False, fill_value='extrapolate')(x)


def rmax_interp(x):
    return -3.5069 + x * 0.42818
def vmax_interp(x):
    return -0.858705 + x * 0.25740337

#  Conversions
SolarMtoGeV = 1.11547 * 10 ** 57.
GeVtoSolarM = 8.96485 * 10 ** -58.
cmtokpc = 3.24078 * 10 ** -22.
kpctocm = 3.08568 * 10 ** 21.
degtorad = np.pi / 180.
radtodeg = 180. / np.pi
newton_G = 4.301 * 10. ** -6  # Units: km^2 kpc / (Solar Mass * s^2)

#  Numerical Quantities -- taken from PDG
hubble = 0.673
H0 = 0.1 * hubble  # Units: km / (s * kpc)
rho_critical = 2.775 * 10 ** 11. * hubble ** 2. * 10. ** -9.  # Units: Solar Mass / (kpc)^3
delta_200 = 200.


def Concentration_parameter(mass, z=0, arxiv_num=13131729, dist=1., vmax=3.):
    """
    Calculates concentration parameter given a subhalo mass and choice of
    paramaterization

    160304057 -- For NFW only
    """
    c = 0.
    if arxiv_num == 13131729:
        if z != 0:
            raise ValueError 
        coeff_array = np.array([37.5153, -1.5093, 1.636 * 10**(-2),
                                3.66 * 10**(-4), -2.89237 * 10**(-5), 
                                5.32 * 10**(-7)])
        for x in range(coeff_array.size):
            c += coeff_array[x] * (np.log(mass * hubble))**x
    elif arxiv_num == 10070438:
        w = 0.029
        m = 0.097
        alpha = -110.001
        beta = 2469.720
        gamma = 16.885
        
        a = w * z - m
        b = alpha / (z + gamma) + beta / (z + gamma)**2.
        
        c = (mass * hubble)**a * 10.**b
    elif arxiv_num == 160106781:
        scale_radius = (10. ** rmax_interp(np.log10(mass)) / 1.21) / 2.163
        c = Virial_radius(mass, m200=True) / scale_radius
    elif arxiv_num == 160304057:
        xsub = dist / 200.
        b = np.array([1., 2., 3.])
        # Mass relation
        a = np.array([-0.195, 0.089, 0.089])
        c = 19.9 * (1. + np.sum((a * np.log(mass * hubble/ 10. ** 8.)) ** b)) *\
            (1. + -0.54 * np.log(xsub))

        # Vmax relation
        #a = np.array([-1.38, 0.83, -0.49])
        #bb = -2.5
        #c = 3.5 * 10. ** 4. * (1. + np.sum((a * np.log(vmax / 10.)) ** b)) *\
        #    (1. + bb * np.log(xsub))
    else:
        print 'Wrong arXiv Number for Concentration Parameter'
    
    return c


def concentrate_in_vmax(vmax, rmax):
    c = 2. * (vmax / (H0 * rmax)) ** 2.
    return c


def rmax_vmax(mass):
    return [10. ** rmax_interp(np.log10(mass)), 10. ** vmax_interp(np.log10(mass))]


def Virial_radius(mass, m200=False):
    """
    Calculates the virial radius of a subhalo by scaleing that of the Milky Way
    :param mass: Subhalo mass in SM
    :return: Virial radius in kpc
    """
    if not m200:
        return 200. * (mass / (1.5 * 10 ** 12.)) ** (1. / 3.)
    else:
        return (3. * mass / (4. * np.pi * rho_critical * delta_200)) ** (1. / 3.)


def Tidal_radius(mass, arxiv=160106781):
    if arxiv == 160106781:
        return 10. ** tidal_interp(np.log10(mass))


def interpola(val, x, y):
    try:
        f = np.zeros(len(val))
        for i, v in enumerate(val):
            if v <= x[0]:
                f[i] = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0])
            elif v >= x[-1]:
                f[i] = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-2])
            else:
                f[i] = interp1d(x, y, kind='cubic').__call__(v)
    except TypeError:
        if val <= x[0]:
            f = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (val - x[0])
        elif val >= x[-1]:
            f = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (val - x[-2])
        else:
            try:
                f = interp1d(x, y, kind='cubic').__call__(val)
            except:
                f = interp1d(x, y, kind='linear').__call__(val)
    return f


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return


class DictDiffer(object):
    """
        Calculate the difference between two dictionaries as:
        (1) items added
        (2) items removed
        (3) keys same in both but changed values
        (4) keys same in both and unchanged values
        """
    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)

    def added(self):
        return self.set_current - self.intersect

    def removed(self):
        return self.set_past - self.intersect

    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] != self.current_dict[o])

    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] == self.current_dict[o])


def adjustFigAspect(fig, aspect=1):

    xsize, ysize = fig.get_size_inches()
    minsize = min(xsize, ysize)
    xlim = .4 * minsize / xsize
    ylim = .4 * minsize / ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)
    return


def str2bool(v):
    if type(v) == bool:
        return v
    elif type(v) == str:
        return v.lower() in ("yes", "true", "t", "1")


def table_gamma_index(annih_prod='BB'):
    """
    Inteded to table the gamma index for large range of mass, so in future we only need
    a simple load and interpoalte funciton.
    :param annih_prod: annihlation products [Currently only works for 'BB']
    """
    #   Don't Use... Pythia8 is giving me a headache. For the moment
    #   I created this file using the PPPC which should give very similar
    #   results -- although apparently there is a bug for c\bar{c}
    #  TODO Create spectrum files for more than just bb
    num_collisions = 10 ** 5.
    integrate_file = MAIN_PATH + "/Spectrum/IntegratedDMSpectrum" + annih_prod + ".dat"
    file_path = MAIN_PATH + "/Spectrum/"
    mass_files = glob.glob(file_path + '*' + annih_prod + '_DMspectrum.dat')
    mass_tab = np.array([])
    for f in mass_files:
        slash_tab = [m.start() for m in re.finditer('/', f)]
        mm = float(f[slash_tab[-1] + 1:f.find('GeV')])
        mass_tab = np.append(mass_tab, [mm])
    gamma_mx = np.zeros(mass_tab.size)
    for i, mx in enumerate(mass_tab):
        print 'Mass: ', mx
        spectrum = np.loadtxt(mass_files[i])
        spectrum[:, 1] = spectrum[:, 1] / num_collisions

        integrated_list = np.loadtxt(integrate_file)
        integrated_rate = interp1d(integrated_list[:, 0], integrated_list[:, 1])

        gamma_list = np.linspace(1.5, 3.0, 100)
        normalize = np.zeros(len(gamma_list))
        for g in range(len(gamma_list)):
            normalize[g] = integrated_rate(mx) / quad(lambda x: x ** (-gamma_list[g]), 1., mx)[0]

        def int_mean_es(x):
            return interpola(x, spectrum[:, 0], spectrum[:, 0] * spectrum[:, 1])
        mean_e = quad(int_mean_es, 1., mx, epsabs=10.**-4., epsrel=10.**-4.)[0]

        mean_e_gam = np.zeros(len(gamma_list))
        for g in range(len(gamma_list)):
            mean_e_gam[g] = quad(lambda x: normalize[g] * x ** (-gamma_list[g] + 1.), 1., mx,
                                 epsabs=10.**-4., epsrel=10.**-4.)[0]

        diff_meanE = np.abs(mean_e_gam - mean_e)
        gamma_mx[i] = gamma_list[diff_meanE.argmin()]

    sv_dir = MAIN_PATH + "/SubhaloDetection/Data/Misc_Items/"
    file_name = 'GammaIndex_given_mx_for_annih_prod_' + annih_prod + '.dat'
    sv_info = np.column_stack((mass_tab, gamma_mx))
    sv_info = sv_info[np.argsort(sv_info[:, 0])]
    np.savetxt(sv_dir + file_name, sv_info)

    return


def integrated_rate_test(mx=100., annih_prod='BB'):
    # This currently doesn't work
    file_path = MAIN_PATH + "/Spectrum/"
    file_path += '{}'.format(int(mx)) + 'GeV_' + annih_prod + '_DMspectrum.dat'

    spectrum = np.loadtxt(file_path)
    imax = 0
    for i in range(len(spectrum)):
        if spectrum[i, 1] < 10 or i == (len(spectrum) - 1):
            imax = i
            break
    spectrum = spectrum[0:imax, :]
    Nevents = 10. ** 5.
    spectrum[:, 1] /= Nevents
    test = interp1d(np.log10(spectrum[:, 0] / mx), np.log10(mx * np.log(10.) * spectrum[:, 1]), kind='cubic', bounds_error=False, fill_value=0.)
    test2 = interp1d(spectrum[:, 0], spectrum[:, 0] * spectrum[:, 1], kind='cubic', bounds_error=False, fill_value=0.)
    e_gamma_tab = np.logspace(0., np.log10(spectrum[-1, 0]), 200)
    print np.column_stack((np.log10(spectrum[:, 0] / mx), np.log10(mx * np.log(10.) * spectrum[:, 1])))
    xtab = np.linspace(np.log10(1. / mx), 0., 200)
    ng2 = np.trapz(10.**test(xtab) / 10. ** xtab, xtab) / np.log(10.)
    mean_e2 = np.trapz(test2(e_gamma_tab), e_gamma_tab)
    rate_interp = UnivariateSpline(spectrum[:, 0], spectrum[:, 1])
    avg_e_interp = UnivariateSpline(spectrum[:, 0], spectrum[:, 0] * spectrum[:, 1])
    num_gamma = rate_interp.integral(1., spectrum[-1, 0])
    mean_e = avg_e_interp.integral(1., spectrum[-1, 0])


    print 'DM Mass: ', mx
    print 'Annihilation Products: ', annih_prod
    print 'Number of Gammas > 1 GeV: ', num_gamma, ng2
    print '<E> Gamma: ', mean_e, mean_e2

    return


def eliminate_redundant_lines(directory):
    files = glob.glob(directory + '/*.dat')
    for f in files:
        print f
        load_f = np.loadtxt(f)
        load_f = np.vstack({tuple(row) for row in load_f})
        load_f = load_f[np.argsort(load_f[:,0])]
        np.savetxt(f, load_f)
    print 'Done!'
    return


def find_max_gen_prof(x, r1, r2):
    return - np.sqrt(x ** (2. - r1) * hyp2f1(3. - r1, r2, 4. - r1, -x) / (3. - r1))


def find_max_KMMDSM_prof(x, gam):
    return - np.sqrt(gamma(3. - gam) * (1. - gammaincc(3. - gam, x)) / x)

