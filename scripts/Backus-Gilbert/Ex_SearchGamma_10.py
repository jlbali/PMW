# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:15:51 2018

@author: mraj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:02 2018

@author: mariela
"""
import numpy as np
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
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



def BG(gamma=0,MaxObs=4,w=0.001):
    print("Computing %.5f"%gamma)
    X.append(gamma)
    BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
    Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    RMSE=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
    print("RMSE: %.5f\n*****************************************"%RMSE)        
    return RMSE
    #return L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    #return -(gamma-.1)*(gamma-.3)       
    
    
    
def calc_gamma(XI,XD,XC,YI,YD,YC,MO):
    End=False
    Error=False
    while not End:
        if (YC>YI)and(YC>YD):
            print("Error")
            Error=True
            End=True
            if YI<YD:
                XF=XI
                YF=YI 
            else:
                XF=XD
                YF=YD 
        elif (YC<YI)and(YC<YD):
            if (np.random.randint(2)==0):
                print("Elijo der")
                Xn=(XC+XD)/2
                Yn=BG(gamma=Xn,MaxObs=MO,w=0.001)
                if Yn<YC:
                    XI=XC
                    XC=Xn
                    YI=YC
                    YC=Yn
                elif (Yn>YD):
                    print("Error")
                    Error=True
                    End=True
                    XF=XC
                    YF=YC
                else:
                    XD=Xn
                    YD=Yn
            else:
                print("Elijo izq")
                Xn=(XC+XI)/2
                Yn=BG(gamma=Xn,MaxObs=MO,w=0.001)
                if Yn<YC:
                    XD=XC
                    XC=Xn
                    YD=YC
                    YC=Yn
                elif (Yn>YI):
                    print("Error")
                    Error=True
                    End=True
                    XF=XC
                    YF=YC
                else:
                    XI=Xn
                    YI=Yn
        else:
            print("Refinar", end=" ")
            if (YD<YI):
                print("a derecha")
                XI=XC
                XC=(XD+XC)/2
                YI=YC
                YC=BG(gamma=XC,MaxObs=MO,w=0.001)
            else:
                print("a izquierda")
                XD=XC
                XC=(XI+XC)/2
                YD=YC
                YC=BG(gamma=XC,MaxObs=MO,w=0.001)
        XF=XC
        YF=YC
                
        D=np.max([np.abs(YI-YC),np.abs(YD-YC),np.abs(XI-XC),np.abs(XD-XC)])
        if D<0.001: 
            End=True
            
    return XF,YF,X,Error

#%%

Bands=['K','KA','KU','X','C']
km=MAPA[4:6]

dic={'KA10':6,'KU10':6,'K10':6,'X10':4,'C10':4,'KA25':20,'KU25':20,'K25':20,'X25':12,'C25':12}

cols = ['MaxObs','km','Band','w','Gamma','Y_gama','X','Error']
df = pd.DataFrame(columns = cols)



#w=0.001

i=0
for band in Bands:
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
    Band= band
    Context=L_Context.Load_Context(wdir)
    L_Syn_Img.Set_Margins(Context)
    Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
    np.random.seed(1)
    TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
    np.random.seed(1)
    TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
    Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)
    
    
    X=[]

    MO=dic[band+km]
    #MO=1
    XI=0
    XD=np.pi/2
    XC=np.pi/4
    
    YI=BG(gamma=XI,MaxObs=MO)
    YD=BG(gamma=XD,MaxObs=MO)
    YC=BG(gamma=XC,MaxObs=MO)
    
    best_gamma = calc_gamma(XI,XD,XC,YI,YD,YC,MO)
    df.loc[i] = [MO,'10',band,0.001,best_gamma[0],best_gamma[1],best_gamma[2],best_gamma[3]]
    i+=1
    
df.to_csv(wdir+'Search_Gamma_10_RMSEV_.csv')

