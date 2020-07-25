# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

from __future__ import print_function

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='LCA_25000m'
MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
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
import matplotlib.pyplot as plt
import L_Files

import copy

#%%
#PROCESS PARAMETERS
NeighborsBehaviour = L_Context.NO_NEIGHBORS
#HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
#HDFfilename='GW1AM2_201406011716_108A_L1SGBTBR_2220220'
#HDFfilename='GW1AM2_201406020358_220D_L1SGBTBR_2220220'
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

Bands=['KA','KU','K','X','C']
col=['r','g','k','y','m']

Respuesta=[]
for MAPA in MAPAS:
    BASE_DIR = my_base_dir.get()
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA

    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context)
    VType=Context['Dict_Vars']['VType']
    L_Syn_Img.Set_Margins(Context)
    
    TbH = L_Syn_Img.Create_Random_Synth_Img(Context,sgm0=5,sgm1=5)
    TbV = L_Syn_Img.Create_Trig_Synth_Img(Context,sgm0=5,sgm1=5)
    ImgSyn = [TbH, TbV]
    titaReal=copy.deepcopy(L_ParamEst.Compute_param(ImgSyn,Context))
    
    priori_factors=np.exp(np.linspace(-10,5,200))
    priori_titas=[copy.deepcopy(L_ParamEst.Compute_param(ImgSyn,Context)) for i in priori_factors]
    for i,tita in enumerate(priori_titas):
        tita['H']['sigma2']*=(priori_factors[i])**2
        tita['V']['sigma2']*=(priori_factors[i])**2
    
    posteriori_titas=[]    
    Band='KA'
    #%%
    Res={'MAPA':MAPA,'titaReal':titaReal}
    for Band in Bands:
        #%%
        print(MAPA, Band)
        Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
        Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1)
        K=Observation['Wt']
        posteriori_titas=[]
        l=len(priori_titas)
        ll=[]
        RSE=[]
        R=np.zeros([4,l])
        for i,tita in enumerate(priori_titas):
            Sol=L_Solvers.Solve(Context,Observation,"Rodgers", tita, obsstd=1.0, verbose=False)
            titaSol=L_ParamEst.Compute_param(Sol,Context)
            posteriori_titas.append(titaSol)
            RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=None, Regular=None, LT=None)
            print("Todo:", RMSE, end=', ')
            R[0,i]=RMSE        
            RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=False, Regular=None, LT=None)
            print("SinMargen:", RMSE, end=', ')
            R[1,i]=RMSE
            RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=False, Regular=None, LT=0)
            print("Agua:", RMSE, end=', ')
            R[2,i]=RMSE
            RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=False, Regular=None, LT=1)
            print("Tierra:", RMSE, end=', ')
            R[3,i]=RMSE
            print(1*'*')
            
            ONE=0*TbH+1
            mu=ONE*((VType==0)*tita['H']['mu'][0]+(VType==1)*tita['H']['mu'][1])
            std=ONE*np.sqrt((VType==0)*tita['H']['sigma2'][0]+(VType==1)*tita['H']['sigma2'][1])
            logstd=np.log(std).sum()
            ll.append(-np.linalg.norm((Sol[0]-mu)/std)**2/2-logstd)
            difH=K.dot(Sol[0])-Observation['Tb']['H']
            RSE.append(-np.linalg.norm(difH)**2/2)

            
        Ori=[np.sqrt(prior_tita['H']['sigma2'][1]) for prior_tita in priori_titas]
        Obt=[np.sqrt(poste_tita['H']['sigma2'][1]) for poste_tita in posteriori_titas]     
        Res[Band]=(R,Ori,Obt,np.array(ll),np.array(RSE))
    Respuesta.append(copy.deepcopy(Res))
#%%
plt.figure(0)
plt.clf()
plt.figure(1)
plt.clf()
plt.figure(2)
plt.clf()
plt.figure(3)
plt.clf()
plt.figure(4)
plt.clf()
for j in range(3):
    Res=Respuesta[j]    
    MAPA=Res['MAPA']
    titaReal=Res['titaReal']
    sigmaOrig=np.sqrt(titaReal['H']['sigma2'][1])
    for i,Band in enumerate(Bands):
        R, Ori, Obt, ll, RSE = Res[Band]
        LOri=np.log(Ori/sigmaOrig)
        LObt=np.log(Obt/sigmaOrig)
        plt.figure(0)
        plt.plot(LOri,LOri,c='b')
        plt.plot(LOri,LObt,c=col[i])
        
        plt.figure(1)
        plt.plot(np.log(priori_factors),R.T[:,1],c=col[i])

        plt.figure(2)
        plt.plot(np.log(priori_factors),ll+RSE/2,c=col[i])
        
        plt.figure(3)
        plt.plot(np.log(priori_factors),ll,c=col[i])
        
        plt.figure(4)
        plt.plot(np.log(priori_factors),RSE/2,c=col[i])
#%%
    j=1
    Band='KA'
    Res=Respuesta[j]
    MAPA=Res['MAPA']
    titaReal=Res['titaReal']
    sigmaOrig=np.sqrt(titaReal['H']['sigma2'][1])
    R, Ori, Obt = Res[Band]
    LOri=np.log(Ori/sigmaOrig)
    LObt=np.log(Obt/sigmaOrig)

    ONE=0*TbH+1
    mu=ONE*((VType==0)*tita['H']['mu'][0]+(VType==1)*tita['H']['mu'][1])
    std=ONE*np.sqrt((VType==0)*tita['H']['sigma2'][0]+(VType==1)*tita['H']['sigma2'][1])
    l=np.log(std).sum()
    (-np.linalg.norm((Sol[0]-mu)/std)**2-l)
    #%%
            dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'H','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,'RMSE':str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)),'CoefCorr':Coef_correlacion(TbH,Sol[0]),'time':t,'Mu_real_0':titaReal['H']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['H']['sigma2'][0]),'Mu_real_1':titaReal['H']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['H']['sigma2'][1]),'Mu_rec_0':tita['H']['mu'][0],'Sigma_rec_0':np.sqrt(tita['H']['sigma2'][0]),'Mu_rec_1':tita['H']['mu'][1],'Sigma_rec_1':np.sqrt(tita['H']['sigma2'][1])}
            df = df.append(dic, ignore_index=True)
            
            dic = {'CellSize':km,'Method': Method,'Band':Band,'Pol': 'V','Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std,'RMSE':str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)),'CoefCorr':Coef_correlacion(TbV,Sol[1]),'time':t,'Mu_real_0':titaReal['V']['mu'][0],'Sigma_real_0':np.sqrt(titaReal['V']['sigma2'][0]),'Mu_real_1':titaReal['V']['mu'][1],'Sigma_real_1':np.sqrt(titaReal['V']['sigma2'][1]),'Mu_rec_0':tita['V']['mu'][0],'Sigma_rec_0':np.sqrt(tita['V']['sigma2'][0]),'Mu_rec_1':tita['V']['mu'][1],'Sigma_rec_1':np.sqrt(tita['V']['sigma2'][1])}
            df = df.append(dic, ignore_index=True)
    
            print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
            if export_solutions:
                L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_'))
    
    
#%%                     

#%%
print(MAPA)
Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1)
VType=Context['Dict_Vars']['VType']

for i,tita in enumerate(priori_titas):
    Sol=L_Solvers.Solve(Context,Observation,"Rodgers", tita, obsstd=1.0, verbose=False)
    titaSol=L_ParamEst.Compute_param(Sol,Context)
    posteriori_titas.append(titaSol)

    


    RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=None, Regular=None, LT=None)
    print("Todo:", RMSE, end=', ')
    R[0,i]=RMSE        
    RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=False, Regular=None, LT=None)
    print("SinMargen:", RMSE, end=', ')
    R[1,i]=RMSE
    RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=False, Regular=None, LT=0)
    print("Agua:", RMSE, end=', ')
    R[2,i]=RMSE
    RMSE, ro = L_Syn_Img.COMP_COND(ImgSyn[0], Sol[0], Context, Margin=False, Regular=None, LT=1)
    print("Tierra:", RMSE, end=', ')
    R[3,i]=RMSE
    print(1*'*')



#%%   

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
#tita=L_ParamEst.recompute_param(ImgSyn,Observation,tita)
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
