import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/SGRB_Tt_EV_thetajet/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/SGRB_Tt_EV_thetajet_k4/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/SGRB_Tt_EV_thetajet_k1/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/SGRB_Tt_EV/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/LGRB_golden_Tt_EV_k2/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/LGRB_golden_Tt_EV_thetajet/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/LGRB_golden_Tt_EV_thetajet_k4/'
path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/LGRB_golden_Tt_EV_thetajet_k4_new/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/LGRB_golden_Tt_EV/'
#path='/Users/giulia/ANALISI/SHALLOWPHASE/DAINOTTI_DALLOSSO/xshallowphase/output/LGRB_golden_Tt_EV_Pfree/'

# pulisce i plot e resetta
plt.clf()
plt.close()

fileinold=path+"oldmodel.txt"
fileinewa=path+"newmodel_alphafix.txt"

#fileinold=path+"oldmodel_E0.01.txt"
#fileinewa=path+"newmodel_alphafix_E0.01.txt"

pthreshold=0.005
print ''
print 'Threshold for P-value: '+str(pthreshold)
print ''

dataold=pd.read_csv(fileinold,comment='!', sep=',', header=None,skiprows=[0],skip_blank_lines=True)
dataold_array=dataold.values
databadpiold=dataold_array[np.where(dataold_array[:,11]<pthreshold)]
datagoodpiold=dataold_array[np.where(dataold_array[:,11]>=pthreshold)]
B_old=datagoodpiold[:,5]
P_old=datagoodpiold[:,7]
k_old=datagoodpiold[:,3]
grbold=databadpiold[:,0].tolist()
print 'GRB with bad fit using old model:', len(grbold), grbold
print ''

datanewa=pd.read_csv(fileinewa,comment='!', sep=',', header=None,skiprows=[0],skip_blank_lines=True)
datanewa_array=datanewa.values
databadpinewa=datanewa_array[np.where(datanewa_array[:,12]<pthreshold)]
datagoodpinewa=datanewa_array[np.where(datanewa_array[:,12]>=pthreshold)]
B_newa=datagoodpinewa[:,6]
P_newa=datagoodpinewa[:,8]
k_newa=datagoodpinewa[:,4]
grbnewa=databadpinewa[:,0].tolist()
print 'GRB with bad fit using new model with a=1.0:', len(grbnewa), grbnewa
print ''

#badgrb=set(grbold).intersection(grbnew,grbnewa)
badgrb=set(grbold).intersection(grbnewa)
print ''
print 'GRB with bad fit using all models are ',len(badgrb),' for a total of '+str(len(dataold_array))+' GRB'
print badgrb
print ''






# PLOT histograms

plt.figure(1)
plt.subplot(311)

#plt.hist(B_old,5,histtype=u'step',color='red',label='B old')
#plt.hist(B_new,5,histtype=u'step',color='blue',label='B new')
plt.hist(B_newa,5,histtype=u'step',color='green',label='B new_a1')

plt.title(path[74:])
plt.xlabel("B[10^14 Gauss]")
plt.ylabel("N")
plt.legend()

plt.subplot(312)
#plt.hist(P_old,5,histtype=u'step',color='red',label='P old')
#plt.hist(P_new,5,histtype=u'step',color='blue',label='P new')
plt.hist(P_newa,5,histtype=u'step',color='green',label='P new_a1')

#plt.title(path[74:])
plt.xlabel("P[ms]")
plt.ylabel("N")
plt.legend()

plt.subplot(313)
#plt.hist(k_old,5,histtype=u'step',color='red',label='k old')
#plt.hist(k_new,5,histtype=u'step',color='blue',label='k new')
plt.hist(k_newa,5,histtype=u'step',color='green',label='k new_a1')

#plt.title(path[74:])
plt.xlabel("k")
plt.ylabel("N")
plt.legend()

plt.show()


if os.path.isfile(path+'check.png'):
	os.system('rm '+path+'check.png')
if not os.path.isfile(path+'check.png'):
    plt.savefig(path+'check.png')
