"""
A module to fit GRB light curves

Require python 3
"""


def info():
    """
    Print some infos
    """
    print("GFIT: a module to fit GRB light curves")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpmath import *
import scipy.special
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.stats as stats
import os

# Constants
PI = 3.14

# --- Constants
msun = 1.989                        # units of 10^33 gr
G=6.67*1.e-8                        # Gravitational constant
c = 2.99792458                      # units of 10^10 cm/s
r0 = 1.2                            # units of 10 km (10^6 cm)
M=1.4
be=G*msun*1.e7*M/(r0*c**2)          # compactness
Ine=(0.247+0.642*be+0.466*be**2)*msun*M*r0**2  # from Lattimer & Prakash
alphax=0.9                          # Alpha parameter of new model

# Range for the data
E01=0.3
E02=10.0
# Range for the computation
E1=1.0
E2=10000.

# --- Initial values of model parameters
B = 25.0                            # Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
spini = 5.                          # initial spin period in units of ms >1ms
omi = 2.0*np.pi/spini               # initial spin frequency 2pi/spini = 6.28 10^3 Hz
k=0.4                               # k=4*epsilon_e kind of radiative efficiency

def read():
    '''
    Reads data from X-ray afterglow luminosity light curves
    '''
    pass

startTxrt = 100.

fb = 1
E0 = 1
def model_old(logt,k,B,omi):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection
    (Dall'Osso et al. 2011) assuming pure magnetic dipole, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        [E 0 is fixed and is initial ejecta energy 0.7*omi^2 10^51 erg]

    NOTA: nelle funzioni che definiscono i modelli, tutti i parametri liberi vanno espressi esplicitamente

    Usage: model_old(logt,k,B,omi)
    """
    t=10.**logt

    a1=2.*(Ine*c**3*1.e5/r0**6)/(B**2.*omi**2.)
    t00=startTxrt # will be in the function
    Ein=0.5*Ine*omi**2          # --> 0.7*omi**2   #10^51 erg
    Li=Ein/a1                 # Li=(0.7*B**2*omi**4)/(3.799*10**6)

    hg1_old=fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t00/a1))
    hg2_old=fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t/a1))

    f_old=(k/t)*(1./(1. + k))*t**(-k)*(E0*t00**k + E0*t00**k - Li*t00**(1+k)*hg1_old + Li*t**(1. + k)*hg2_old)

    return np.log10(f_old)

def plot_model_old(tmin=-1., tmax=6.):
    time = np.logspace(tmin, tmax, num=100, base=10.0)

    plt.plot(np.log10(time),model_old(np.log10(time),k,B,omi),'r--',label='D11 model')
    plt.xlabel('time from trigger [s]')
    plt.ylabel('log Luminosity x 10^51 [erg s^-1]')
    plt.legend()
    plt.show()

def model_ax(logt,k,B,omi):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection
    introducing the Contopoulos and Spitkovsky 2006 formula assuming alphax, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        [E 0 is fixed to 10^51 erg]
    """
    t=10**logt
    t00=startTxrt

    hg1_a=fb*scipy.special.hyp2f1((alphax-2.)/(alphax-1.), 1. + k, 2.+k, 3.72092e-7*(alphax-1.)*(omi**2)*(B**2)*t00)
    hg2_a=fb*scipy.special.hyp2f1((alphax-2.)/(alphax-1.), 1.+k, 2.+k,3.72092e-7*(alphax-1.)*(omi**2)*(B**2)*t)
    f_ax=(k/t)*(1/(1 + k))*(2.77055e-7)*(t**(-k))*(3.6094*10**6*E0*t00**k + 3.6094*10**6*E0*k*t00**k + (B**2)*(omi**4)*(t**(1.+k))*hg2_a - (B**2)*(omi**4)*t00**(1 + k)*hg1_a)

    return np.log10(f_ax)

def plot_model_ax(tmin=-1., tmax=6.):
    time = np.logspace(tmin, tmax, num=100, base=10.0)

    plt.plot(np.log10(time),model_ax(np.log10(time),k,B,omi),'b--',label='CS06 model')
    plt.xlabel('time from trigger [s]')
    plt.ylabel('log Luminosity x 10^51 [erg s^-1]')
    plt.legend()
    plt.show()
