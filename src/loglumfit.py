
"""
    Author: Giulia Stratta
This script
 1. Reads data from X-ray afterglow luminosity light curves
 2. If 'EV' is present in the output dir, then correction for cosmological evolution is applied
    If 'jet' is present in the output dir, then correction for beaming angle is applied to data
    If 'isoNS' is present in the output dir, then correction for beaming angle is applied to model
    (in this last case we assume that the magnetar emits isotropically)
 3. Defines model: L(t)=kE(t)/t (from Simone Notebook)
 4. Fits model to the data and compute best fit param and cov. matrix
 5. Plots best fit model on data
 6. Saves plots and best fit parameters in ../output

NOTE: E0 is not computed but fixed.

To run the script:
    ipython --pylab
    run loglumfit.py

Requested input: name of GRB (e.g. 060605A), jet angle in deg, E0 in 10^52 erg unit

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


# pulisce i plot e resetta eventuali residui del run precedente
plt.clf()
plt.close()

# definisce il GRB di cui voglio il fit
fi = raw_input(' grb light curve file [e.g. 050603]: ') or "050603"


"""
1. SET OUTPUT FILES
"""
path1="./output/LGRB_golden_Tt_EV_thetajet_k4_new/"

# Definisce i file di output
outfileold_single=path1+"oldmodel_"+fi+".txt"
outfilenewx_single=path1+"newmodel_alphafix_"+fi+".txt"

# Create the header for the old model
if os.path.isfile(outfileold_single):
	os.system('rm '+outfileold_single)
if not os.path.isfile(outfileold_single):
	os.system('touch '+outfileold_single)
	out_file = open(outfileold_single,"a")
	out_file.write("GRB,Tstart,E051,k,dk,B14,dB14,Pms,dPms,chi2,dof,p-val"+"\n")
	out_file.close()

# Create the header for the new model
if os.path.isfile(outfilenewx_single):
	os.system('rm '+outfilenewx_single)
if not os.path.isfile(outfilenewx_single):
	os.system('touch '+outfilenewx_single)
	out_file = open(outfilenewx_single,"a")
	out_file.write("GRB,Tstart,E051,alphax,k,dk,B14,dB14,Pms,dPms,chi2,dof,p-val"+"\n")
	out_file.close()

"""
2. SET INPUT CONSTANTS
"""
# Definisce la banda di energia [E1,E2] in keV a cui voglio la luminosita

# Range for the data
E01=0.3
E02=10.0
# Range for the computation
E1=1.0
E2=10000.

# set decimal places
mp.dps = 25
# Setting the mp.pretty option will use the str()-style output for repr() as well
mp.pretty = True

# --- Constants
msun = 1.989                        # units of 10^33 gr
G=6.67*1.e-8                        # Gravitational constant
c = 2.99792458                      # units of 10^10 cm/s
r0 = 1.2                            # units of 10 km (10^6 cm)
M=1.4
be=G*msun*1.e7*M/(r0*c**2)          # compactness
Ine=(0.247+0.642*be+0.466*be**2)*msun*M*r0**2  # from Lattimer & Prakash
#Ine = 0.35*msun*1.4*(r0)**2         # 1.4 10^45 gr cm^2
alphax=0.9                          # Alpha parameter of new model

# --- Initial values of model parameters
B = 25.0                            # Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
spini = 5.                          # initial spin period in units of ms >1ms
omi = 2.0*np.pi/spini               # initial spin frequency 2pi/spini = 6.28 10^3 Hz
k=0.4                               # k=4*epsilon_e kind of radiative efficiency

"""
 4. READ DATA
"""

#path = '/Users/giovanni/Works/GRB/plateau_fit/giuia_code/data/TimeLuminosityLC/'
path = '/Users/giovanni/Works/GRB/GRB_curve_fit/TimeLuminosityLC/'

filename=path+fi+'TimeLuminosityLC.txt'
data=pd.read_csv(filename,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)

# "trasforma" i dati in un ndarray
data_array = data.values

# elimina le righe con NaN alla prima colonna
table    = data_array[np.logical_not(np.isnan(data_array[:,0])),:]
dlogtime = table[:,1]
dloglum  = table[:,3]

# --- Calcola il fattore correttivo per portare la lum in banda 1-10000 keV

# Read beta and Tstart from file (please check)
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
Kcorr = (E2**(1.-beta+1.e-8) - E1**(1.-beta+1.e-8))/(E02**(1.-beta+1.e-8) - E01**(1.-beta+1.e-8))

# --- Calcola il fattore correttivo per l'evoluzione cosmologica

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
thetajetstr=raw_input('theta_jet [90.0]:') or "90.0"
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

E0=float(raw_input('set E051 (es. 1):'))

"""
5. PLOT DATA POINT
"""
En1=str(E1)
En2=str(E2)
plt.title('GRB'+fi+' ['+En1+'-'+En2+' keV]')

plt.plot(ltime,llum,'.', label='data',color='b')
plt.errorbar(ltime, llum, yerr=dllum,fmt='none',ecolor='b')
plt.xlabel('time from trigger [s]')
plt.ylabel('log Luminosity x 10^51 [erg s^-1]')
plt.show()

print("Start time in sec from file:", startTxrtFromFile) # (see beta_parameters_and_Tt.dat)
startTxrt_str = raw_input(' Start time in sec: ') or float(startTxrtFromFile)
startTxrt=float(startTxrt_str)
plt.axvline(x=np.log10(startTxrt),color='k', linestyle='--')
plt.show()


# ToDo Write one function for each model

"""
 3. DEFINE MODELS

 Simone's NOTE:

 spini = initial spin period [in millisecond],
 B = magnetic dipole field [in units of 10^(14)G],
 r0 = typical NS radius [in units of 10 km],
 Ine = moment of inertia of the NS [in units of 10^(45)].

 In principle, all of these have values that are not well determined.
 HOWEVER:
 A) r0 and Ine have only minor uncertainties. The NS radius is assumed here to be 12 km, hence the radius is always written as 1.2.
 The coefficient 0.35 in the moment of inertia is somewhat depenedent on the NS EOS.
 Both the coefficient 1.2 and 0.35 can only vary in a restricted range, and are considered as given constants here.
 B) A different argument holds for spini AND B. These are actual free parameters, that should be determined by the fits.
 In this first part I fix their values in order to produce plots of the relevant quantities,
 that will be needed to compare between the "old", magnetic dipole formula,
 and the "new" formula, proposed by Contopoulos Spitkovsky 2006).
 (*ADDITIONAL UNITS: tsdi is in units of seconds; Ein,the initial spin energy, is in units of 10^(51) ergs.)

"""

def model_old(logt,k,B,omi):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection
    (Dall'Osso et al. 2011) assuming pure magnetic dipole, as function of
        k = radiative efficiency (0.3)
        B = magnetic field (5. in units of 10^14 Gauss)
        omi = initial spin frequency (2pi)
        [E 0 is fixed and is initial ejecta energy 0.7*omi^2 10^51 erg]

    NOTA: nelle funzioni che definiscono i modelli, tutti i parametri liberi vanno espressi esplicitamente

    Usage: model_old()
    """
    t=10.**logt

    a1=2.*(Ine*c**3*1.e5/r0**6)/(B**2.*omi**2.)
    t00=startTxrt # will be in the function
    Ein=0.5*Ine*omi**2          # --> 0.7*omi**2   #10^51 erg
    Li=Ein/a1                 # Li=(0.7*B**2*omi**4)/(3.799*10**6)

    hg1_old=fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t00/a1))
    hg2_old=fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t/a1))

    f_old=(k/t)*(1./(1. + k))*t**(-k)*(E0*t00**k + E0*k*t00**k - Li*t00**(1+k)*hg1_old + Li*t**(1. + k)*hg2_old)
    return np.log10(f_old)



def model_ax(logt,k,B,omi):
    """
    Description: Energy evolution inside the external shock as due to radiative losses+energy injection introducing the Contopoulos and Spitkovsky 2006 formula assuming alphax, as function of
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


t0=np.logspace(-1.,7., num=100, base=10.0)
t1=t0[np.where(t0>startTxrt)]
t=t1[np.where(t1<10**(ltime[-1]))]


"""
6. DEFINE FITTING FUNCTIONS
"""


# FIT MODEL ON DATA
#http://www2.mpia-hd.mpg.de/~robitaille/PY4SCI_SS_2014/_static/15.%20Fitting%20models%20to%20data.html

# initial model
#plt.loglog(txrt, model_a05(txrt,k,B,omi,E0),'k--',label='start model')

# old model (2011 paper)
#plt.loglog(t, model_old(t,0.66,12.2,2*np.pi/1.18,1.04),'k--',label='start model')


# NOTA: Fissare Tstart significa fissare anche E0
# E0 puo essere indeterminato se Ein e molto maggiore
# E0/T0 = lumin. senza magnetar
# se Lin>>E0/T0 allora E0 non riesce ad essere determinare

# fitta un modello tra i 2 definiti sui dati logaritmici txrt lxrt

def fitmodelold(model, x, y, dy):
    # Define initial parameters inside the function
    p0=np.array([k,B,omi])
    #popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.0001,1.e-3,0.05], [4.0, 1000.,105.]),maxfev=10000)
    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.0001,1.e-3,0.05], [4.0, 1000.,105.]))
    print "------ "
    print " k [",k,"] =", "%.5f" %popt[0], "+/-", "%.5f" %pcov[0,0]**0.5
    print " B [",B,"(10^14 G)]  =", "%.5f" %popt[1], "+/-", "%.5f" %pcov[1,1]**0.5
    print " omi [2pi/Pspin_i=",omi,"(10^3 Hz)] =", "%.5f" %popt[2], "+/-", "%.5f" %pcov[2,2]**0.5
    print " Spin Period P [ms]=",2.0*np.pi/popt[2], "+/-",2.0*np.pi*(pcov[2,2]**0.5)/(popt[2]**2.0)
    print " E0 fixed [(10^51 erg)] =", E0
    print " L(Tt)=",model(np.log10(startTxrt),popt[0],popt[1],popt[2])
    print " E0=(L(Tstart))*Tstart/k=",(10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2])))*startTxrt/popt[0]
    print "------  "

    E051=(10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2])))*startTxrt/popt[0]
    Pms=2.0*np.pi/popt[2]
    dPms=Pms*pcov[2,2]**0.5/popt[2]
    print 'Pms, dPms = ', Pms, dPms

    ym=model(x,popt[0],popt[1],popt[2])
    print stats.chisquare(f_obs=y,f_exp=ym)
    mychi=sum(((y-ym)**2)/dy**2)
    #mychi=sum(((y-ym)**2)/ym)
    dof=len(x)-len(popt)
    print "my chisquare=",mychi
    print "dof=", dof
    p_value = 1.-stats.chi2.cdf(x=mychi,df=dof)
    print "P value",p_value

    bfmodel=model(np.log10(t),popt[0],popt[1],popt[2])

    out_file = open(outfileold_single,"a")
    out_file.write(fi+","+str(startTxrt)+","+str(E051)+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %Pms)+","+str("%.5f" %dPms)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
    out_file.close()
    # ToDo Return the data
    return plt.plot(np.log10(t), bfmodel,'r',label='D11 (p-val='+str("%.3f" %p_value)+')')



def fitmodelnewx(model, x, y, dy):
    p0=np.array([k,B,omi])
    #popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.0001,1.e-3,0.05], [4.0, 1000.,105.]),maxfev=10000)
    popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.0001,1.e-3,0.05], [4.0, 1000.,105.]))
    print "------ "
    print "k [",k,"] =", "%.5f" %popt[0], "+/-", "%.5f" %pcov[0,0]**0.5
    print "B [",B,"(10^14 G)]  =", "%.5f" %popt[1], "+/-", "%.5f" %pcov[1,1]**0.5
    print "omi [2pi/spin_i=",omi,"(10^3 Hz)] =", "%.5f" %popt[2], "+/-", "%.5f" %pcov[2,2]**0.5
    print " Spin Period [ms]=",2.0*np.pi/popt[2], "+/-",2.0*np.pi/popt[2]*(pcov[2,2]**0.5)/popt[2]
    print " E0 [fixed (10^51 erg)] =", E0
    print "alpha (fixed) =", alphax
    print " E051=(L(Ttstart))*Tstart/k=",10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2]))*startTxrt/popt[0]
    print "------  "

    E051=10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2]))*startTxrt/popt[0]
    Pms=2.0*np.pi/popt[2]
    dPms=Pms*(pcov[2,2]**0.5)/popt[2]

    ym=model(x,popt[0],popt[1],popt[2])
    print stats.chisquare(f_obs=y,f_exp=ym)
    mychi=sum(((y-ym)**2)/dy**2)
    #mychi=sum(((y-ym)**2)/ym)
    dof=len(x)-len(popt)
    print "my chisquare=",mychi
    print "dof=", dof
    p_value = 1.-stats.chi2.cdf(x=mychi,df=dof)
    print "P value",p_value

    bfmodel=model(np.log10(t),popt[0],popt[1],popt[2])

    out_file = open(outfilenewx_single,"a")
    out_file.write(fi+","+str(startTxrt)+","+str(E051)+","+str(alphax)+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %Pms)+","+str("%.5f" %dPms)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
    out_file.close()

    return plt.plot(np.log10(t), bfmodel,'c',label='CS06 alpha='+str(alphax)+' (p-val='+str("%.3f" %p_value)+')')


"""
6. FIT AND PLOT INITIAL MODELS
"""

plt.plot(np.log10(t),model_old(np.log10(t),k,B,omi),'r--',label='D11 (initial)')
plt.plot(np.log10(t),model_ax(np.log10(t),k,B,omi),'c--',label='CS06 alpha='+str(alphax)+' (initial)')
plt.show()

logstartTxrt=np.log10(startTxrt)
txrt=ltime[np.where(ltime>logstartTxrt)]
lxrt=llum[np.where(ltime>logstartTxrt)]
dlxrt=dllum[np.where(ltime>logstartTxrt)]

print ' '
print 'model_old'
fitmodelold(model_old,txrt,lxrt,dlxrt)

print ' '
print 'model_a with alpha fixed'
fitmodelnewx(model_ax,txrt,lxrt,dlxrt)

# --- Salva il plot nella directory output

plt.legend()
plt.show()
plt.savefig(path1+fi+'_new.png')
