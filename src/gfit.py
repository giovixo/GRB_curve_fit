## alphax=0.9                          # Alpha parameter of new model


# >>> Parameters for the K correction (to move in the right block)
# Range for the data
# Per convertire i dati nella luminosita' bolometrica (a cui si riferisce il modello)
## E01=0.3
## E02=10.0
# Range for the computation
## E1=1.0
## E2=10000.
# Beta letto dalla tabella dei GRB

# --- Initial values of model parameters
## B = 25.0                            # (*) Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
## spini = 5.                          # (*) initial spin period in units of ms >1ms
## omi = 2.0*np.pi/spini               # (*) initial spin frequency 2pi/spini = 6.28 10^3 Hz
## k=0.4                               # (*) k=4*epsilon_e kind of radiative efficiency

## E0 = 1                              # frozen (in the function)

# -- ... --
## fb = 1
# Tempo di inizio del plateau (dal file beta...)
## startTxrt = 100.

"""
A module to fit GRB light curves
"""

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

# --- Constants
msun = 1.989                        # units of 10^33 gr
G=6.67*1.e-8                        # Gravitational constant
c = 2.99792458                      # units of 10^10 cm/s
r0 = 1.2                            # units of 10 km (10^6 cm)
M=1.4
be=G*msun*1.e7*M/(r0*c**2)          # compactness
Ine=(0.247+0.642*be+0.466*be**2)*msun*M*r0**2  # from Lattimer & Prakash

# --- Initial values of model parameters
B = 25.0                            # Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
spini = 5.                          # initial spin period in units of ms >1ms
omi = 2.0*np.pi/spini               # initial spin frequency 2pi/spini = 6.28 10^3 Hz
k=0.4                               # k=4*epsilon_e kind of radiative efficiency

def info():
    """
    Print some infos
    """
    print("GFIT: a module to fit GRB light curves")

def read(fi='050603', Tt=0, thetajetstr="90"):
    '''
    Reads data from X-ray afterglow luminosity light curves

    Return (data_array, startTxrtFromFile, fb)

    Usage:
    (data, time, fb) = read(fi='050603', Tt=0, thetajetstr="90")
    '''
    # Path for the output (it is used as a flag in this function)
    path1="./output/LGRB_golden_Tt_EV_thetajet_k4_new/"

    # Define the filename
    path = '/Users/giovanni/Works/GRB/GRB_curve_fit/TimeLuminosityLC/'
    filename = path + fi + 'TimeLuminosityLC.txt'
    # Read the data
    data=pd.read_csv(filename,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)
    # Data cleaning
    data_array = data.values
    table    = data_array[np.logical_not(np.isnan(data_array[:,0])),:]
    dlogtime = table[:,1]
    dloglum  = table[:,3]

    # Read beta and Tstart from file
    filein=path+'beta_parameters_and_Tt.dat'
    dataparam=pd.read_csv(filein,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)
    dataparam_array=dataparam.values
    print "Index for ",fi," : ", np.where(dataparam_array==fi)
    datafi=dataparam_array[np.where(dataparam_array[:,0]==fi)]
    datafiflat=datafi.flatten()
    z=float(datafiflat[1])
    beta=float(datafiflat[2])
    dbeta=float(datafiflat[3])
    Tt=float(datafiflat[5])

    # luminosity correction from E01,E02 to E1,E2 (see Dainotti et al 2013)
    # Range for the data
    E01=0.3
    E02=10.0
    # Range for the computation
    E1=1.0
    E2=10000.
    Kcorr = (E2**(1.-beta+1.e-8) - E1**(1.-beta+1.e-8))/(E02**(1.-beta+1.e-8) - E01**(1.-beta+1.e-8))

    # Correct the luminosity and time for cosmological EVoloution
    if 'EV' in path1:
        loglum=table[:,2]-51+np.log10(Kcorr)+0.05*np.log10(1+z)
        logtime=table[:,0]+0.85*np.log10(1+z)
    else:
        loglum=table[:,2]-51+np.log10(Kcorr)
        logtime=table[:,0]

    #   Correct also the Tstart time for cosmological EV.
    if Tt > 0.0 :
        if 'EV' in path1:
            startTxrtFromFile=(10**Tt)/((1.+z)*(1+z)**(-0.85))
        else:
            startTxrtFromFile=(10**Tt)/(1.+z)
    if Tt == 0.0 :
        startTxrtFromFile=0.0

    # --- Corregge i dati ed il modello per il beaming del getto (see theta_jet_only.dat)
    # 90 means NOT collimated
    # thetajetstr=raw_input('theta_jet [90.0]:') or "90.0"
    thetajrad=float(thetajetstr)*(np.pi/180.0)
    #logaritmo del fattore di collimazione
    f=np.log10((1-np.cos(thetajrad)))

    # sort times and riassess lum vector
    index=np.argsort(logtime)
    # ---
    ltime=logtime[index] # my x
    llum=loglum[index]+f # my y
    dllum=dloglum[index] # my err_y (+f or NOT +f?)

    # Discussion?
    if 'isoNS' in path1:
        fb=10.0**f
        print '!! jet beaming correction applied to model with fb= ,'+str(fb)+' !!'
    else:
        fb=1.0
        print '!! No jet beaming correction applied to model with fb=,'+str(fb)+' !!'

    return (data_array, startTxrtFromFile, fb)


def model_old(logt, k, B, omi, E0=1, fb=1, startTxrt=100):
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
    t = 10.**logt

    a1 = 2.*(Ine*c**3*1.e5/r0**6)/(B**2.*omi**2.)
    t00 = startTxrt
    Ein = 0.5*Ine*omi**2          # --> 0.7*omi**2   #10^51 erg
    Li = Ein/a1                 # Li=(0.7*B**2*omi**4)/(3.799*10**6)

    hg1_old = fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t00/a1))
    hg2_old = fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t/a1))

    f_old = (k/t)*(1./(1. + k))*t**(-k)*(E0*t00**k + E0*t00**k - Li*t00**(1+k)*hg1_old + Li*t**(1. + k)*hg2_old)

    return np.log10(f_old)

def plot_model_old(tmin=-1., tmax=6.):
    time = np.logspace(tmin, tmax, num=100, base=10.0)

    plt.plot(np.log10(time),model_old(np.log10(time),k,B,omi),'r--',label='D11 model')
    plt.xlabel('time from trigger [s]')
    plt.ylabel('log Luminosity x 10^51 [erg s^-1]')
    plt.legend()
    plt.show()

def model_ax(logt, k, B, omi, E0=1, fb=1, startTxrt=100, alphax=0.9):
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
