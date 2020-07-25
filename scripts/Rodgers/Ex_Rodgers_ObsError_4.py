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
Methods=['STAT', 'TYCH']
Bands=['KA','KU','K', 'X','C']
Pols=['H','V']
cols=['Method','Band','Pol','RMSE','STD_0','STD_1','ObsStdErr']

Observations={}
Context=L_Context.Load_Context(wdir)
L_Syn_Img.Set_Margins(Context)
for Band in Bands:
    Observations[Band], tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
#%%
    
    
#df=pd.DataFrame(columns=cols)

def write_log(d,fname='logfile.log'):
    with open(fname, "a") as myfile:
         myfile.write(d)
         myfile.write("\n")
            

def add_line(l,d):
    write_log(str(d),'Logfile.log')
    l.append(d)
#%%
    
l=[]
d={'Method':'TYCH','Band':'KA','Pol':'H','RMSE':1.234,'STD_0':2.345,'STD_1':3.456,'ObsStdErr':1.0}
#add_line(l,d)

#df=pd.DataFrame(l,columns=cols)
#%%

BG_param={
'H':{'w': {'KA':0.001,'KU':0.001,'K':0.001, 'X':0.001,'C':0.001},   
     'g': {'KA':0.7,'KU':0.7,'K':0.7, 'X':0.7,'C':0.7},   
     'MO':{'KA':20,'KU':20,'K':20, 'X':12,'C':12}},
'V':{'w': {'KA':0.001,'KU':0.001,'K':0.001, 'X':0.001,'C':0.001},   
     'g': {'KA':0.7,'KU':0.7,'K':0.7, 'X':0.7,'C':0.7},   
     'MO':{'KA':20,'KU':20,'K':20, 'X':12,'C':12}}
}

#%%
TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=10,sgm1=10)
TbV = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=5,sgm1=5)
Band='KA'
tita=L_ParamEst.recompute_param([TbH,TbV],Observations[Band],tita)


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
        Sol=L_Solvers.Solve(Context,Observations[Band], "Global_GCV_Tichonov", tita, damp=std*std)

        RMSEH=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
        RMSEV=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)    
        print(" -done")
        print("Ty_RMSEsHw: %.3f"%RMSEH)
        print("Ty_RMSEsVw: %.3f"%RMSEV)
        tita=L_ParamEst.recompute_param(Sol,Observations[Band],tita)        
        
        D[Band]['Ty_RMSEHs'].append(RMSEH)
        D[Band]['Ty_RMSEVs'].append(RMSEV)
        D[Band]['Ty_STDHo0'].append(np.sqrt(tita['H']['sigma2'][0]))
        D[Band]['Ty_STDHo1'].append(np.sqrt(tita['H']['sigma2'][1]))
        D[Band]['Ty_STDVo0'].append(np.sqrt(tita['V']['sigma2'][0]))
        D[Band]['Ty_STDVo1'].append(np.sqrt(tita['V']['sigma2'][1]))

#%%
d={'STDs':D['ObsErr'][0:164]}
for Band in Bands:
    d['ST_STDHo0_'+Band]=D[Band]['STDHo0'][0:164]
    d['ST_STDHo1_'+Band]=D[Band]['STDHo1'][0:164]
    d['ST_RMSEHs_'+Band]=D[Band]['RMSEHs'][0:164]
    d['ST_STDVo0_'+Band]=D[Band]['STDVo0'][0:164]
    d['ST_STDVo1_'+Band]=D[Band]['STDVo1'][0:164]
    d['ST_RMSEVs_'+Band]=D[Band]['RMSEVs'][0:164]
    d['TY_STDHo0_'+Band]=D[Band]['Ty_STDHo0'][0:164]
    d['TY_STDHo1_'+Band]=D[Band]['Ty_STDHo1'][0:164]
    d['TY_RMSEHs_'+Band]=D[Band]['Ty_RMSEHs'][0:164]
    d['TY_STDVo0_'+Band]=D[Band]['Ty_STDVo0'][0:164]
    d['TY_STDVo1_'+Band]=D[Band]['Ty_STDVo1'][0:164]
    d['TY_RMSEVs_'+Band]=D[Band]['Ty_RMSEVs'][0:164]
#%%
len(D['ObsErr'])
for Band in Bands:
    Band
    len(D[Band]['STDHo0'])
    len(D[Band]['STDHo1'])
    len(D[Band]['RMSEHs'])
    len(D[Band]['STDVo0'])
    len(D[Band]['STDVo1'])
    len(D[Band]['RMSEVs'])
    len(D[Band]['Ty_STDHo0'])
    len(D[Band]['Ty_STDHo1'])
    len(D[Band]['Ty_RMSEHs'])
    len(D[Band]['Ty_STDVo0'])
    len(D[Band]['Ty_STDVo1'])
    len(D[Band]['Ty_RMSEVs'] )   
    
    #%%
    
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
for Band in ['KU','C']:#Bands:
    plt.plot(df['STDs'],df['ST_RMSEHs_'+Band],label='ST_RMSE '+Band)
    plt.plot(df['STDs'],df['TY_RMSEHs_'+Band],label='TY_RMSE '+Band)
#plt.legend(loc='lower right')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.show()
plt.savefig('2OE_'+MAPA+"H_RMSE")
#%%

plt.clf()
plt.title("STD in vs STD out (STAT) H. Different std and bands. "+MAPA)
plt.xlabel('obs std err')
plt.ylabel('std out')

plt.plot(df['STDs'],df['STDs'])
for Band in Bands:
    plt.plot(df['STDs'],df['ST_STDHo0_'+Band],label='ST_'+Band)
    plt.plot(df['STDs'],df['TY_STDHo0_'+Band],label='TY_'+Band)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
plt.show()
plt.savefig('2OE_'+MAPA+"H_STD")

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
for Band in ['K','KU']:#Bands:
    plt.plot(df['STDs'],df['ST_RMSEVs_'+Band],label='ST_RMSE '+Band)
    plt.plot(df['STDs'],df['TY_RMSEVs_'+Band],label='TY_RMSE '+Band)

plt.legend(loc='upper right')
plt.show()
plt.savefig('3OE_'+MAPA+"V_RMSE")
#%%
plt.clf()
plt.title("STD in vs STD out (STAT) V. Different std and bands. "+MAPA)
plt.xlabel('std in')
plt.ylabel('std out')

plt.plot(df['STDs'],df['STDs'])
for Band in ['K','KU']:#Bands:
    plt.plot(df['STDs'],df['ST_STDVo0_'+Band],label='ST_STD '+Band)
    plt.plot(df['STDs'],df['TY_STDVo0_'+Band],label='TY_STD '+Band)

#for Band in Bands:
#    plt.plot(df['STDs'],df['STDVo0_'+Band],label=Band)
plt.legend(loc='lower center', bbox_to_anchor=(0, 0.05),
          ncol=1, fancybox=True, shadow=True)
plt.show()
plt.savefig('3OE_'+MAPA+"V_STD")

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
    
