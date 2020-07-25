# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

from __future__ import print_function

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_25km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA


#import __builtin__
#__builtin__.ML = []
#global ML
#ML=[]

#Import libraries
import L_ReadObs
import L_Context
import L_Output
import L_Solvers
import L_Syn_Img
import L_ParamEst
import numpy as np
from scipy.sparse.linalg import lsqr

import copy

#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
#HDFfilename='GW1AM2_201406011716_108A_L1SGBTBR_2220220'
HDFfilename='GW1AM2_201406020358_220D_L1SGBTBR_2220220'
#HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

Band='X'

#Method="Weights"
#Method="LSQR"
#Method="Global_GCV_Tichonov"

#%%
def Create_Random_Synth_Img(Context,sgm=10):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    VType=Context['Dict_Vars']['VType']
    nlt=VType.max()+1

    Vars_of_type=[]
    for lt in range(nlt):
        Vars_of_type.append(np.where(VType==lt)[0])

    nvt=list(map(len,Vars_of_type))        
    mu=[190,271]
    sigma=[sgm,sgm]
    
    
    Tb=np.zeros(n_Vars)
    for lt in range(nlt):
        Tb[Vars_of_type[lt]]=np.random.normal(mu[lt],sigma[lt],nvt[lt])
    return Tb
#%%
def main(): 
    #%%###################
    ## PREPROCESSING    
    #%%
    #Load Context  
    damp=1.0
    Context=L_Context.Load_Context(wdir)

    #Create Synthetic Image 
    TbH=Create_Random_Synth_Img(Context,sgm=10)
    TbV=Create_Random_Synth_Img(Context,sgm=.6)
    
    #[5,2,1,...]
    #Export Synthetic Image Map
    #L_Output.Export_Solution([TbH,TbV], Context, Band, "SynthImg")

    #Load ellipses and compute synthetic observation
    Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename, Context, Band) #Load Observations & Kernels

    Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=damp) #Simulate SynObs
    #print("LandType approximation error:%.2f, %.2f"%(Observation['LSQRSols']['H']['norm'],Observation['LSQRSols']['V']['norm']))
    #Observation['LSQRSols']={'H':lsqr(M,Observation['Tb']['H']),'V':lsqr(M,Observation['Tb']['V'])}
     

    
    #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
    L_Syn_Img.Set_Margins(Context)
    titaOrig=copy.deepcopy(L_ParamEst.recompute_param([TbH,TbV],Observation,tita))
    L_ParamEst.pr(titaOrig,'Synthetic Image')
    
    K=Observation['Wt']
    Tbh=Observation['Tb']['H']
    Tbv=Observation['Tb']['V']
    VType=Observation['VType']

    
    #%%
    STDs=np.exp(np.arange(-10,8.001,.1))#[np.exp(-10),np.exp(-5),np.exp(-4),np.exp(-2),np.exp(-1),np.exp(0),np.exp(1),np.exp(2),np.exp(3),np.exp(4),np.exp(5)]#,3,5,7,9,11,13,15]
    STDh0=[]
    STDv0=[]
    STDh1=[]
    STDv1=[]
    RSEh=[]
    PEh=[]
    RSEv=[]
    PEv=[]
    LLh=[]
    LLv=[]
    
    ll=[]
    for std in STDs:
        tita=copy.deepcopy(titaOrig)
        for pol in ['H','V']:
            for lt in [0,1]:
                tita[pol]['sigma2'][lt]=std*std
        
        Sol=L_Solvers.Solve(Context,Observation,"Rodgers", tita,damp=damp)
        
        tita=copy.deepcopy(titaOrig)
        for pol in ['H','V']:
            for lt in [0,1]:
                tita[pol]['sigma2'][lt]=std*std
        difH=Tbh-K.dot(Sol[0])
        difV=Tbv-K.dot(Sol[1])
        #sn=np.sqrt(Tbh.shape[0])
        RSEh.append(-np.linalg.norm(difH)**2)
        RSEv.append(-np.linalg.norm(difV)**2)
        err_str="RMSE, H:%.2f, V:%.2f "%(RSEh[-1],RSEv[-1])
        #RSE.append([RSEh,RSEv])
        ONE=0*TbH+1
        mu=ONE*((VType==0)*tita['H']['mu'][0]+(VType==1)*tita['H']['mu'][1])
        std=ONE*np.sqrt((VType==0)*tita['H']['sigma2'][0]+(VType==1)*tita['H']['sigma2'][1])
        l=np.log(std).sum()
        PEh.append(-np.linalg.norm((Sol[0]-mu)/std)**2-l)
        mu=ONE*((VType==0)*tita['V']['mu'][0]+(VType==1)*tita['V']['mu'][1])
        std=ONE*np.sqrt((VType==0)*tita['V']['sigma2'][0]+(VType==1)*tita['V']['sigma2'][1])
        l=np.log(std).sum()
        PEv.append(-np.linalg.norm((Sol[1]-mu)/std)**2-l)
        LLh.append(RSEh[-1]+PEh[-1])
        LLv.append(RSEv[-1]+PEv[-1])
        #PE.append([PEh,PEv])
        
        tita=L_ParamEst.recompute_param(Sol,Observation,tita)        
        STDh0.append(np.sqrt(tita['H']['sigma2'][0]))
        STDh1.append(np.sqrt(tita['H']['sigma2'][1]))
        STDv0.append(np.sqrt(tita['V']['sigma2'][0]))
        STDv1.append(np.sqrt(tita['V']['sigma2'][1]))
        
        L_ParamEst.pr(tita,'Reconstructed Image')

    #%%
    import matplotlib.pyplot as plt
    plt.clf()
   
    llplot(STDs,STDs)
    llplot(STDs,STDh0)
    llplot(STDs,STDh1)
    llplot_diag_crux(np.sqrt(titaOrig['H']['sigma2'][0]),e=2)
    llplot_diag_crux(np.sqrt(titaOrig['H']['sigma2'][1]),e=2)

    plt.figure()
    llplot(STDs,STDs)
    llplot(STDs,STDv0)
    llplot(STDs,STDv1)
    llplot_diag_crux(np.sqrt(titaOrig['V']['sigma2'][0]),e=2)
    llplot_diag_crux(np.sqrt(titaOrig['V']['sigma2'][1]),e=2)
    #%
    plt.figure()
    lplot(STDs,PEh)
    lplot(STDs,RSEh)
    lplot(STDs,LLh)
    #%
    plt.figure()
    lplot(STDs,PEv)
    lplot(STDs,RSEv)
    lplot(STDs,LLv)
    
    
    #%%
    plt.figure()
    llplot(STDs,STDs)
    llplot(STDs,STDh0)
    lplot(STDs,[l/10000 for l in LLh])
#%%
LLh=[PEh[i]-RSEh[i] for i in range(len(PEh))]
LLv=[PEv[i]-RSEv[i] for i in range(len(PEh))]
#%%
main()

#%%
def lplot(x,y):
    plt.plot(np.log(x),y)

def llplot(x,y):
    plt.plot(np.log(x),np.log(y))
    
def llplot_diag_crux(x,e):
    llplot([x/e,x*e],[x*e,x/e])
##%%
###····test de bias····###
#Context=L_Context.Load_Context(wdir)
#
##Create Synthetic Image 
#TbH=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=10)
#TbV=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=5)
#
##Load ellipses and compute synthetic observation
#Observation, tita=L_ReadObs.Load_PKL_Obs(HDFfilename, Context, Band) #Load Observations & Kernels
#Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0) #Simulate SynObs
#
##Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
#L_Syn_Img.Set_Margins(Context)
#
#tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
#Sol=L_Solvers.Solve(Context,Observation,"Rodgers_IT", tita)
#print("RMSE H: %.3f V: %.3f "%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context),L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
#
#import copy
#tita_IT=copy.deepcopy(tita)
#
##%%
#import numpy as np
#F= np.arange(1.0,1.5,.05)
#for f in F:
#    tita=copy.deepcopy(tita_IT)
#    for pol in ['H','V']:
#        for lt in [0,1]:
#            tita[pol]['sigma2'][lt]*=f
#    Sol=L_Solvers.Solve(Context,Observation,'Rodgers', tita)
#    print("Factor: %.3f RMSE H: %.3f V: %.3f "%(f,L_Syn_Img.RMSE_M(TbH,Sol[0],Context),L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
#
#            
#
#            
#
#
#
