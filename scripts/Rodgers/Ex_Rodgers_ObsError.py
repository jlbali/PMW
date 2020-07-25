# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:15:19 2017

@author: rgrimson
"""
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_25km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
#Import libraries
import copy
import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_ParamEst
import L_Syn_Img
import numpy as np

import pandas as pd
import time

import matplotlib.pyplot as plt
# INICIALIZO LOS VALORES A EXPERIMENTAR:
#%%
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Bands=['KA','KU','K', 'X','C']
#Bands=['KA','K','C']
Observations={}
Context=L_Context.Load_Context(wdir)
L_Syn_Img.Set_Margins(Context)

D={}
for Band in Bands:
    Observations[Band], tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
    D[Band]={}
    D[Band]['RMSEHs']=[]
    D[Band]['RMSEVs']=[]
    D[Band]['STDHo0']=[]
    D[Band]['STDHo1']=[]
    D[Band]['STDVo0']=[]
    D[Band]['STDVo1']=[]

D['STDHi0']=[]
D['STDHi1']=[]
D['STDVi0']=[]
D['STDVi1']=[]
D['ObsErr']=[]
    

#%%
TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=10,sgm1=5)
TbV = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=5,sgm1=5)
Band='KA'
tita=L_ParamEst.recompute_param([TbH,TbV],Observations[Band],tita)
D['STDHi0'].append(np.sqrt(tita['H']['sigma2'][0]))
D['STDHi1'].append(np.sqrt(tita['H']['sigma2'][1]))
D['STDVi0'].append(np.sqrt(tita['V']['sigma2'][0]))
D['STDVi1'].append(np.sqrt(tita['V']['sigma2'][1]))


STDs=[10**i for i in np.arange(-1,0.71,.01)]
for std in STDs:
    D['ObsErr'].append(std)
    for Band in Bands:
        Observations[Band]=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observations[Band],Obs_error_std=std)
        L_ParamEst.compute_param([TbH,TbV], Observations[Band])
        #STAT
        Sol=L_Solvers.Solve_Rodgers_IT_NB(Context,Observations[Band], tita, obsvar=std*std,tol=0.00005, max_iter=50)

        RMSEH=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
        RMSEV=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
        print(" -done")
        print("RMSEsHw: %.3f"%RMSEH)
        print("RMSEsVw: %.3f"%RMSEV)
        tita=L_ParamEst.recompute_param(Sol,Observations[Band],tita)        
        
        D[Band]['RMSEHs'].append(RMSEH)
        D[Band]['RMSEVs'].append(RMSEV)
        D[Band]['STDHo0'].append(np.sqrt(tita['H']['sigma2'][0]))
        D[Band]['STDHo1'].append(np.sqrt(tita['H']['sigma2'][1]))
        D[Band]['STDVo0'].append(np.sqrt(tita['V']['sigma2'][0]))
        D[Band]['STDVo1'].append(np.sqrt(tita['V']['sigma2'][1]))

        #Tycvh
#        Sol=L_Solvers.Solve(Context,Observations[Band], tita, obsvar=std*std,tol=0.00005, max_iter=50)
#
#        RMSEH=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
#        RMSEV=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
#        print(" -done")
#        print("RMSEsHw: %.3f"%RMSEH)
#        print("RMSEsVw: %.3f"%RMSEV)
#        tita=L_ParamEst.recompute_param(Sol,Observations[Band],tita)        
#        
#        D[Band]['RMSEHs'].append(RMSEH)
#        D[Band]['RMSEVs'].append(RMSEV)
#        D[Band]['STDHo0'].append(np.sqrt(tita['H']['sigma2'][0]))
#        D[Band]['STDHo1'].append(np.sqrt(tita['H']['sigma2'][1]))
#        D[Band]['STDVo0'].append(np.sqrt(tita['V']['sigma2'][0]))
#        D[Band]['STDVo1'].append(np.sqrt(tita['V']['sigma2'][1]))

#%%
d={'STDs':STDs}
for Band in Bands:
    d['STDHo0_'+Band]=D[Band]['STDHo0']
    d['STDHo1_'+Band]=D[Band]['STDHo1']
    d['RMSEHs_'+Band]=D[Band]['RMSEHs']
    d['STDVo0_'+Band]=D[Band]['STDVo0']
    d['STDVo1_'+Band]=D[Band]['STDVo1']
    d['RMSEVs_'+Band]=D[Band]['RMSEVs']
df=pd.DataFrame(d)
df.to_csv('Eval_Rodgers_ObsErr_25km.csv')

#%%
#aux=df.sort(['STDHi0'])
plt.figure(1)
plt.clf()
plt.title("STAT H RMSE. Different std and bands. "+MAPA)
plt.xlabel('std for each land type')
plt.ylabel('RMSE')
plt.plot(df['STDs'],df['STDs'])
for Band in Bands:
    plt.plot(df['STDs'],df['RMSEHs_'+Band],label='RMSE '+Band)
plt.legend(loc='lower right')
plt.show()
plt.savefig('OE_'+MAPA+"H_RMSE")
#%%

plt.clf()
plt.title("STD in vs STD out (STAT) H. Different std and bands. "+MAPA)
plt.xlabel('obs std err')
plt.ylabel('std out')

plt.plot(df['STDs'],df['STDs'])
for Band in Bands:
    plt.plot(df['STDs'],df['STDHo0_'+Band],label=Band)

plt.legend(loc='upper right')
plt.show()
plt.savefig('OE_'+MAPA+"H_STD")

#%%
for Band in Bands:
    plt.clf()
    plt.title("STD in vs STD out (both LT) and RMSE. "+MAPA+". Band: "+Band)
    plt.xlabel('std err in')
    plt.ylabel('std out/RMSE')
    plt.plot(df['STDs'],df['STDs'])
    plt.plot(df['STDs'],df['STDHo0_'+Band],label='LT=0')
    plt.plot(df['STDs'],df['STDHo1_'+Band],label='LT=1')
    #plt.plot(df['STDVi0'],df['STDVo0_'+Band])
    #plt.plot(df['STDVi1'],df['STDVo1_'+Band])
    plt.plot(df['STDs'],df['RMSEHs_'+Band],label='RMSE')
    #plt.plot(df['STDs'],df['RMSEVs_'+Band])
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig('OE_'+MAPA+"H_STD_"+Band)
    
#%%
plt.clf()
plt.title("STAT V RMSE. Different std and bands. "+MAPA)
plt.xlabel('std obs err')
plt.ylabel('RMSE')
plt.plot(df['STDs'],df['STDs'])
for Band in Bands:
    plt.plot(df['STDs'],df['RMSEVs_'+Band],label='RMSE '+Band)
plt.legend(loc='upper right')
plt.show()
plt.savefig('OE_'+MAPA+"V_RMSE")
#%%
plt.clf()
plt.title("STD in vs STD out (STAT) V. Different std and bands. "+MAPA)
plt.xlabel('std in')
plt.ylabel('std out')

plt.plot(df['STDs'],df['STDs'])
for Band in Bands:
    plt.plot(df['STDs'],df['STDVo0_'+Band],label=Band)

plt.legend(loc='upper right')
plt.show()
plt.savefig('OE_'+MAPA+"V_STD")

#%%
for Band in Bands:
    plt.clf()
    plt.title("STD in vs STD out (both LT) and RMSE. "+MAPA+". Band: "+Band)
    plt.xlabel('std err in')
    plt.ylabel('std out/RMSE')
    plt.plot(df['STDs'],df['STDs'])
    plt.plot(df['STDs'],df['STDVo0_'+Band],label='LT=0')
    plt.plot(df['STDs'],df['STDVo1_'+Band],label='LT=1')
    #plt.plot(df['STDVi0'],df['STDVo0_'+Band])
    #plt.plot(df['STDVi1'],df['STDVo1_'+Band])
    plt.plot(df['STDs'],df['RMSEVs_'+Band],label='RMSE')
    #plt.plot(df['STDs'],df['RMSEVs_'+Band])
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('OE_'+MAPA+"V_STD_"+Band)
    
