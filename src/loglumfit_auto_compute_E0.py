
"""
    Author: Giulia Stratta

This script
 1. Reads data from X-ray afterglow luminosity light curve and correct for jet beaming angle
 2. Defines model: ejecta energy variation with time assuming energy injection (from Simone Notebook)
 3. Fits model to the data and compute best fit param and cov. matrix
 4. Plots best fit model on data
 5. Saves plot in ../output

 NOTA: "compute_E0" means that a first guess of E0 is computed and written
 into files with extension _E0.txt, that are then read by loglumfit_auto.py to get
 the new initial energy E0 for each GRB.
 E0 is initially fixed to 10^51 erg into each model
 and a new E0 is computed from the model (lum*T/k).

 NOTA2: if no jet beaming angle is wanted, then remove "f" additional factor from llum

 Per lanciare lo script:
    ipython --pylab
    run lumfit

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


"""
1. SET INPUT
"""

path = '/Users/giovanni/Works/GRB/plateau_fit/giuia_code/data/TimeLuminosityLC/'
#fi = raw_input(' grb light curve file [e.g. 050603]: ') or "050603"
f=open(path+'Long_GoldenGRB.dat')
#f=open(path+'Long_GoldenGRB_EV.dat')
#f=open(path+'LongGRB.dat')
#f=open(path+'ShortGRB.dat')
#f=open(path+'testgrb.dat')
datagrb=f.read()
f.close()


"""
 set OUTPUT FILES
"""
#path1="./output/LGRB_Tt/"
#path1="./output/LGRB_golden_Tt_EV/"
#path1="./output/LGRB_golden_Tt_03_100keV/"
#path1="./output/LGRB_golden_Tt_EV_thetajet/"
#path1="./output/LGRB_golden_Tt_EV_thetajet_k4/"
path1="./output/LGRB_golden_Tt_EV_thetajet_k4_new/"
#path1="./output/LGRB_golden_Tt/"

#path1="./output/SGRB_Tt/"
#path1="./output/SGRB_Tt_EV_thetajet/"
#path1="./output/SGRB_Tt_EV_thetajet_k4/"
#path1="./output/test/"

# Definisce i file di output dove scrivo E0 ottenuto con la prima iterazione
# fissando prima E0=10^51, calcolando il modelloe poi E0=L*T/k
outfileold=path1+"oldmodel_E0.txt"
outfilenewx=path1+"newmodel_alphafix_E0.txt"
#outfileold=path1+"oldmodel.txt"
#outfilenew=path1+"newmodel.txt"
#outfilenewx=path1+"newmodel_alphafix.txt"

if os.path.isfile(outfileold):
    os.system('rm '+outfileold)
if os.path.isfile(outfilenewx):
    os.system('rm '+outfilenewx)

# If outfileold and ouflienewx do not exist then:
if not os.path.isfile(outfileold):
#    os.system('touch '+path1+'oldmodel_E0.txt')
    os.system('touch '+outfileold)
    out_file = open(outfileold,"a")
    out_file.write("GRB,Tstart,E051,k,dk,B14,dB14,Pms,dPms,chi2,dof,p-val"+"\n")
    out_file.close()

if not os.path.isfile(outfilenewx):
#    os.system('touch '+path1+'newmodel_alphafix_E0.txt')
    os.system('touch '+outfilenewx)
    out_file = open(outfilenewx,"a")
    out_file.write("GRB,Tstart,E051,alphax,k,dk,B14,dB14,Pms,dPms,chi2,dof,p-val"+"\n")
    out_file.close()


E0old=0.1
E0newa=0.1



print '  !!!!!!!!'
print ''
print ' This code computes the most likely E051 for each GRB read in input from: '
print ''
print '  Input file:',path+str(f)
print ''
print '...and writes E0 into a file with suffix _E0 in the output dir:'
print ''
print '  Output:',path1
print ''
print '  !!!!!!!!'
print ''
check=str(raw_input('Ok with input file and output dir? If not, exit now!'))



"""
2. SET INPUT CONSTANTS
"""

# Definisce la banda di energia [E1,E2] in keV a cui voglio la luminosita
E1=1.0
E2=10000.0
#E1=0.3
#E2=100.0
# qui la banda di energia [E01,E02] in keV a cui ho la luminosita
E01=0.3
E02=10.0

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
alphax=0.9


# --- Initial values of model parameters
B = 5.0                            # Magnetic field in units of 10^14 Gauss= 1/(gr*cm*s)
spini = 1.                          # initial spin period in units of ms >1ms
omi = 2.0*np.pi/spini               # initial spin frequency 2pi/spini = 6.28 10^3 Hz
k=0.4                               # k=4*epsilon_e kind of radiative efficiency
#mu32=(B*r0**3)/2                   # magnetic dipole in unit 10^32 Gauss cm^-3

# --- useful formulas to remind
#Ein = 0.5*Ine*omi**2                            # initial spin energy 27.7 10^51 erg
#tsdi = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5   # Initial spin down time for the standard magnetic dipole formula 3.799*10^6/B2*omi**2 s
#Li=Ein/tsdi                                     # Initial spindown lum. (?) 0.007 10^51 erg/s
#a1 = 3*Ine*c**3/(B**2*(r0)**6*omi**2)*10**5     # tsdi
# ----

# --- in the Notebook but not used in the code
#Lni=Ein/(2./3.*a1)
#tsd=a1*2./3.

#a1=tsdi
#a2=2./(2.-alpha)*(2./3.*a1)
# ----

"""
 4. READ DATA
"""

listgrb=datagrb.split()
for fi in listgrb:

    # --- Legge le curve di luce nella dir TimeLuminosityLC
    filename=path+fi+'TimeLuminosityLC.txt'
    data=pd.read_csv(filename,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)
    # trasforma i dati in un ndarray
    data_array=data.values
    # elimina le righe con NAN alla prima colonna
    table=data_array[np.logical_not(np.isnan(data_array[:,0])),:]

    dlogtime=table[:,1]
    dloglum=table[:,3]

    # --- Calcola il fattore correttivo per portare la lum in banda E1-E2

    # Read z, beta and Tstart from file
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

    # Read theta_jet from file
    fileinjet=path+'theta_jet_only.dat'
    datathetaj=pd.read_csv(fileinjet,comment='!', sep='\t', header=None,skiprows=None,skip_blank_lines=True)
    datathetaj_array=datathetaj.values
    print "Index for ",fi," : ", np.where(datathetaj_array==fi)
    datafij=datathetaj_array[np.where(datathetaj_array[:,0]==fi)]
    datafijflat=datafij.flatten()
    thetajetstr=datafijflat[1]

    # luminosity correction from E01,E02 to E1,E2
    Kcorr = (E2**(1.-beta+1.e-8) - E1**(1.-beta+1.e-8))/(E02**(1.-beta+1.e-8) - E01**(1.-beta+1.e-8))

    if 'EV' in path1:
        loglum=table[:,2]-51+np.log10(Kcorr)+0.05*np.log10(1+z)
        logtime=table[:,0]+0.85*np.log10(1+z)
    else:
        loglum=table[:,2]-51+np.log10(Kcorr)
        logtime=table[:,0]

    #   Correct also the Tstart time for cosmological ev.
    if Tt > 0.0 :
        if 'EV' in path1:
            print '!! Cosmological evolution correction applied !!'
            startTxrtFromFile=(10**Tt)/((1.+z)*(1+z)**(-0.85))
        else:
            print '!! No cosmological evolution correction applied !!'
            startTxrtFromFile=(10**Tt)/(1.+z)
    if Tt == 0.0 :
        startTxrtFromFile=0.0

    print("Start time in sec from file:", startTxrtFromFile)
    # If the read start time is ok, just retype it, otherwise type the right start time
    # startTxrt = float(raw_input(' Start time in sec: '))
    startTxrt = float(startTxrtFromFile)


    # --- Corregge i dati ed il modello per il beaming del getto

    thetajrad=float(thetajetstr)*(np.pi/180.0)
    #logaritmo del fattore di collimazione
    f=np.log10((1-np.cos(thetajrad)))


    # sort times and riassess lum vector
    index=np.argsort(logtime)
    ltime=logtime[index]

    if 'isoNS' in path1:
        fb=10.0**f
        print '!! jet beaming correction applied to model with fb= ,'+str(fb)+' !!'
    else:
        fb=1.0
        print '!! No jet beaming correction applied to model with fb=,'+str(fb)+' !!'


    if 'jet' in path1:
        print '!! jet beaming correction applied to data !!'
        llum=loglum[index]+f
    else:
        print '!! No jet beaming correction applied to data !!'
        llum=loglum[index]

    dllum=dloglum[index]

    print fi, thetajetstr, f, fb

    # --- Plot data points
    En1=str(E1)
    En2=str(E2)
    plt.title('GRB'+fi+' ['+En1+'-'+En2+' keV]')

    #plt.plot(logtime,loglum, color='r')
    #plt.errorbar(logtime,loglum,yerr=dloglum,color='r')
    plt.plot(ltime,llum,'.', label='data',color='b')
    plt.errorbar(ltime, llum, yerr=dllum,fmt='none',ecolor='b')
    plt.xlabel('time from trigger [s]')
    plt.ylabel('log Luminosity x 10^51 [erg s^-1]')
    plt.show()

    # Se uso il Notebook
    #%matplotlib inline
    #plt.loglog(time,lum,'.')
    #plt.xlabel('time from trigger [s]')
    #plt.ylabel('Luminosity x 10^51 [erg cm^-2 s^-1]')
    #plt.ylabel('0.3-10 keV flux [erg cm^-2 s^-1]')
    #plt.show()



    # If the read start time is ok, just retype it, otherwise type the right start time
    # startTxrt = float(raw_input(' Start time in sec: '))
    startTxrt = float(startTxrtFromFile)

    # Create a time vector with which plot the model
    #t0=np.linspace(100.0,1.0e6,10000)
    t0=np.logspace(-1.,7., num=100, base=10.0)
    t1=t0[np.where(t0>startTxrt)]
    t=t1[np.where(t1<10**(ltime[-1]))]



    """
     DEFINE MODELS

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
    # I modelli li devo definire in funzione di logt per poter fare il fit

    # --- Useful formula to remind but not used in the code
    #Lsdold1=Ein/(tsdi*(1 + t/tsdi)**2)  # pure dipole radiation spin down lum.
    #Lsdold2= Li/(1 + t/a1)**2           # stessa formula di Lsdold1 ma scritta in modo piu semplice

    # Initial spin down time for the new formula by C&S06
    #tsdnew= (2./3.)*(3*Ine*c**3*10.**5/(r0**6))/(B**2. * omi**2.)   # tsdnew=(2/3)tsdi
    #tsdnew= 2./3.*3.7991745075226948*10**6./(B**2. * omi**2.)
    #ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha)/(2 - alpha))

    # new spin down lum. expression for alpha2
    #ls=r0**6/(4*c**3*10**5)*B**2*omi**4/(1 + (2 - alpha2)/4*r0**6/(Ine*c**3*10**5)*B**2*omi**2*t)**((4 - alpha2)/(2 - alpha2))
    # ----

    def model_old(logt,k,B,omi):
        """
        Description: Energy evolution inside the external shock as due to radiative losses+energy injection
        (Dall'Osso et al. 2011) assuming pure magnetic dipole, as function of
            k = radiative efficiency (0.3)
            B = magnetic field (5. in units of 10^14 Gauss)
            omi = initial spin frequency (2pi)
            [E 0 is fixed to 10^51 erg]

        NOTA: nelle funzioni che definiscono i modelli, tutti i parametri liberi vanno espressi esplicitamente

        Usage: model_old()
        """
        t=10**logt

        # tsdi= (3*Ine*c**3*10**5/(r0)**6)*(1/(B**2*omi**2))
        # a1=tsdi
        a1=2.*(Ine*c**3*1.e5/r0**6)/(B**2.*omi**2.)
        t00=startTxrt
        # Li=Ein/tsdi  Initial spindown lum. (?) 0.007 10^51 erg/s
        Ein=0.5*Ine*omi**2          # --> 0.7*omi**2   #10^51 erg
        Li=Ein/a1                 # Li=(0.7*B**2*omi**4)/(3.799*10**6)

        hg1_old=fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t00/a1))
        hg2_old=fb*scipy.special.hyp2f1(2., 1. + k, 2. + k, -(t/a1))
        #f_old=(k/t)*(1./(1. + k))*t**(-k)*(E0old*t00**k + E0old*k*t00**k + Li*t**(1+k)*hg2_old - Li*t00**(1+k)*hg1_old)
        f_old=(k/t)*(1./(1. + k))*t**(-k)*(E0old*t00**k + E0old*t00**k - Li*t00**(1+k)*hg1_old + Li*t**(1. + k)*hg2_old)
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

        #tsdnew=4/3 tsdi
        t00=startTxrt
        #tsdnew=4/3 tsdi
        #anew=(4./3.)*(3.799*10.**6.)/(B**2.*omi**2.)
        #anew=(2.)*(I*c**3./r0**6)/(B**2.*omi**2.)

        # 1/anew=1.97411*10**(-7)*(B**2*omi**2)

        #hg1_a=scipy.special.hyp2f1((4. - alphax)/(2. -alphax), 1. + k, 2.+k, 1.97411*10**(-7)*(alphax-2.)*B**2*omi**2)
        #hg2_a=scipy.special.hyp2f1((4. -alphax)/(2. -alphax), 1.+k, 2.+k, 1.97411*10**(-7)*(alphax-2.)*B**2*omi**2*t)
        hg1_a=fb*scipy.special.hyp2f1((alphax-2.)/(alphax-1.), 1. + k, 2.+k, 3.72092e-7*(alphax-1.)*(omi**2)*(B**2)*t00)
        hg2_a=fb*scipy.special.hyp2f1((alphax-2.)/(alphax-1.), 1.+k, 2.+k,3.72092e-7*(alphax-1.)*(omi**2)*(B**2)*t)
        f_ax=(k/t)*(1/(1 + k))*(2.77055e-7)*(t**(-k))*(3.6094*10**6*E0newa*t00**k + 3.6094*10**6*E0newa*k*t00**k + (B**2)*(omi**4)*(t**(1.+k))*hg2_a - (B**2)*(omi**4)*t00**(1 + k)*hg1_a)

        return np.log10(f_ax)



    """
    Define FITTING FUNCTIONS
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

        p0=np.array([k,B,omi])
        popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.0001,1.e-3,0.05], [4.0, 1000.,105.]),maxfev=10000)

        #Ein = 0.5*Ine*popt[2]**2                            # initial spin energy 27.7 10^51 erg
        #tsdi = 3*Ine*c**3/(popt[1]**2*(r0)**6*popt[2]**2)*10**5   # Initial spin down time for the standard magnetic dipole formula 3.799*10^6/B2*omi**2 s
        #Li=Ein/tsdi

        print "------ "
        print " k [",k,"] =", "%.5f" %popt[0], "+/-", "%.5f" %pcov[0,0]**0.5
        print " B [",B,"(10^14 G)]  =", "%.5f" %popt[1], "+/-", "%.5f" %pcov[1,1]**0.5
        print " omi [2pi/spin_i=",omi,"(kHz)] =", "%.5f" %popt[2], "+/-", "%.5f" %pcov[2,2]**0.5
        print " Spin Period [ms]=",2.0*np.pi/popt[2], "+/-",2.0*np.pi/popt[2]*(pcov[2,2]**0.5)/popt[2]
        print " E0 (fixed) [10^51 erg) =", E0old
        print " L(Tt)=",10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2]))
        print " E051=(L(Ttstart))*Tstart/k=",(10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2])))*startTxrt/popt[0]
        print "------  "

        E051=(10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2])))*startTxrt/popt[0]
        Pms=2.0*np.pi/popt[2]
        dPms=Pms*(pcov[2,2]**0.5)/popt[2]
        print 'Pms, dPms=', Pms, dPms

    #    ym=model(x,popt[0],popt[1],popt[2],popt[3])
        ym=model(x,popt[0],popt[1],popt[2])
        print stats.chisquare(f_obs=y,f_exp=ym)
        mychi=sum(((y-ym)**2)/dy**2)
        #mychi=sum(((y-ym)**2)/ym)
        dof=len(x)-len(popt)
        print "my chisquare=",mychi
        print "dof=", dof
        p_value = 1-stats.chi2.cdf(x=mychi,df=dof)
        print "P value",p_value

        bfmodel=model(np.log10(t),popt[0],popt[1],popt[2])

        out_file = open(outfileold,"a")
        #out_file.write(fi+","+str(startTxrt)+","+str(E051)+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %popt[2])+","+str("%.5f" %pcov[2,2]**0.5)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
        out_file.write(fi+","+str(startTxrt)+","+str(E051)+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %Pms)+","+str("%.5f" %dPms)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
        out_file.close()

        return plt.plot(np.log10(t), bfmodel,'r',label='D11 (p-val='+str(p_value)+')')



    def fitmodelnewx(model, x, y, dy):

        p0=np.array([k,B,omi])
        popt, pcov = curve_fit(model, x, y, p0, sigma=dy, bounds=([0.0001,1.e-3,0.05], [4.0, 1000.,105.]),maxfev=10000)
        print "------ "
        print " k [",k,"] =", "%.5f" %popt[0], "+/-", "%.5f" %pcov[0,0]**0.5
        print " B [",B,"(10^14 G)]  =", "%.5f" %popt[1], "+/-", "%.5f" %pcov[1,1]**0.5
        print " omi [2pi/spin_i=",omi,"(10^3 Hz)] =", "%.5f" %popt[2], "+/-", "%.5f" %pcov[2,2]**0.5
        print " Spin Period [ms]=",2.0*np.pi/popt[2], "+/-",2.0*np.pi/popt[2]*(pcov[2,2]**0.5)/popt[2]
        print " E0 [fixed (10^51 erg)] =", E0newa
        print " alpha (fixed) =", alphax
        print " L(Tt)=",10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2]))
        print "Tstart=",startTxrt
        print " E051=(L(Ttstart))*Tstart/k=",(10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2])))*startTxrt/popt[0]
        print "------  "

        E051=(10**(model(np.log10(startTxrt),popt[0],popt[1],popt[2])))*startTxrt/popt[0]
        Pms=2.0*np.pi/popt[2]
        dPms=Pms*(pcov[2,2]**0.5)/popt[2]

        ym=model(x,popt[0],popt[1],popt[2])
        print stats.chisquare(f_obs=y,f_exp=ym)
        mychi=sum(((y-ym)**2)/dy**2)
        #mychi=sum(((y-ym)**2)/ym)
        dof=len(x)-len(popt)
        print "my chisquare=",mychi
        print "dof=", dof
        p_value = 1-stats.chi2.cdf(x=mychi,df=dof)
        print "P value",p_value

        bfmodel=model(np.log10(t),popt[0],popt[1],popt[2])

        out_file = open(outfilenewx,"a")
        out_file.write(fi+","+str(startTxrt)+","+str(E051)+","+str(alphax)+","+str("%.5f" %popt[0])+","+str("%.5f" %pcov[0,0]**0.5)+","+str("%.5f" %popt[1])+","+str("%.5f" %pcov[1,1]**0.5)+","+str("%.5f" %Pms)+","+str("%.5f" %dPms)+","+str("%.5f" %mychi)+","+str("%.5f" %dof)+","+str("%.5f" %p_value)+"\n")
        out_file.close()

        return plt.plot(np.log10(t), bfmodel,'c',label='CS06 alpha='+str(alphax)+' (p-val='+str(p_value)+')')


    """
    6. PLOT INITIAL MODELS
    """

    plt.plot(np.log10(t),model_old(np.log10(t),k,B,omi),'r--',label='D11 (initial)')
    plt.plot(np.log10(t),model_ax(np.log10(t),k,B,omi),'c--',label='CS06 alpha='+str(alphax)+' (initial)')
    plt.show()
    #plt.loglog(t,f_a05)
    #plt.loglog(t,f_a1)
    #plt.xlabel('time from trigger [s]')
    #plt.ylabel('Luminosity / 10^51 [erg/s]')

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

    plt.legend()
    plt.show()

# --- Salva il plot nella directory output
    """
    answer=str(raw_input('save plot? [y,n]:'))

    if answer == 'y' :

        if os.path.isfile(path1+fi+'.png'):
    	    os.system('rm '+path1+fi+'.png')
        if not os.path.isfile(path1+fi+'.png'):
            plt.savefig(path1+fi+'.png')

    if answer == 'n' :
        plt.clf()
        plt.close()
    # pulisce i plot e resetta
    """

    if os.path.isfile(path1+fi+'_E0.png'):
        os.system('rm '+path1+fi+'_E0.png')
    if not os.path.isfile(path1+fi+'_E0.png'):
        plt.savefig(path1+fi+'_E0.png')

    plt.clf()
    plt.close()
