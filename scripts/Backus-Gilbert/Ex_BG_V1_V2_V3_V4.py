# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:56:11 2018

@author: mraj
"""

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
import sys

BASE_DIR = my_base_dir.get()
#MAPA='Ajo_25km'
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
import pickle


Band='KA'
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
Context=L_Context.Load_Context(wdir)
L_Syn_Img.Set_Margins(Context)
Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename,Context,Band)
np.random.seed(1)
TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
np.random.seed(1)
TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)

Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1.0)



def BG(gamma=np.pi/4,MaxObs=4,w=0.001,V=1):
    if V==4:
        BG_Coef=L_Solvers.BG_Precomp_MaxObs_V4(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
    elif V==3:
        BG_Coef=L_Solvers.BG_Precomp_MaxObs_V3_b(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
    elif V==2:
        BG_Coef=L_Solvers.BG_Precomp_MaxObs_V2(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
    else:
        BG_Coef=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma,w=w,MaxObs=MaxObs)
    Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    RMSE=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    print("RMSE: %.5f\n*****************************************"%RMSE)        
    return Sol, RMSE
    #return L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    #return -(gamma-.1)*(gamma-.3)       
    

def precomputeBBox(Context, Observation):
        Tb=Observation['Tb']
        K=Observation['Wt']
        VType=Observation['VType']
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        SolH=np.zeros(n_Vars)
        SolV=np.zeros(n_Vars)
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        LTG=Context['Dict_Grids']['LandType_Grid']##
        X=range(CIdG.shape[0])##
        Y=range(CIdG.shape[1])##
        #M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        #A=np.zeros([n_Vars,n_Ells])

        print('Precomputing distances for Backus Gilbert (fine, %d cells)'%n_Vars)
        #print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        distances_matrix_list = []
        J=CIdG*0
        nx,ny=J.shape
        for ID in range(n_Vars):
            print("Preprocesando ", ID, " de ", n_Vars)
            distances_matrix = np.zeros(J.shape)
            [Wx,Wy]=np.where(CIdG==ID)
            mx=Wx.min()
            Mx=Wx.max()
            my=Wy.min()
            My=Wy.max()
            cx=(mx+Mx)/2
            cy=(my+My)/2
            width=Mx-mx
            height=My-my
            area_celda = len(Wx)
            area_bbox = (width+1)*(height+1)            
            #print("Area de celda: ", area_celda, " Area de bbox: ", area_bbox)            
            if area_celda != area_bbox:
                print("Area de celda: ", area_celda, " Area de bbox: ", area_bbox)
            for i in range(nx):
                for j in range(ny):
                    dx = np.max((0.0, np.abs(i-cx) - width/2.0))                    
                    dy = np.max((0.0, np.abs(j-cy) - height/2.0))                    
                    #distances_matrix[i,j]=np.sqrt(dx**2 + dy**2)
                    distances_matrix[i,j]=dx**2 + dy**2
            distances_matrix_list.append(distances_matrix)
            #if ID == 5:
            #    return distances_matrix_list
        return distances_matrix_list

def precompute_all(Context, Observation):
        Tb=Observation['Tb']
        K=Observation['Wt']
        VType=Observation['VType']
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        SolH=np.zeros(n_Vars)
        SolV=np.zeros(n_Vars)
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        LTG=Context['Dict_Grids']['LandType_Grid']##
        X=range(CIdG.shape[0])##
        Y=range(CIdG.shape[1])##
        #M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        #A=np.zeros([n_Vars,n_Ells])

        print('Precomputing distances for Backus Gilbert (fine, %d cells)'%n_Vars)
        #print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        distances_matrix_list = []
        J=CIdG*0
        nx,ny=J.shape
        for ID in range(n_Vars):
            print("Preprocesando ", ID, " de ", n_Vars)
            distances_matrix = np.zeros(J.shape)
            [Wx,Wy]=np.where(CIdG==ID)
            for i in range(nx):
                print("Elemento i: ", i)                
                for j in range(ny):
                    dmin = None
                    for k in range(len(Wx)):
                        puntoX = Wx[k]
                        puntoY = Wy[k]
                        d = (i - puntoX)**2 + (j-puntoY)**2
                        if dmin == None or d < dmin:
                            dmin = d
                    #distances_matrix[i,j]=np.sqrt(dx**2 + dy**2)
                    distances_matrix[i,j]=dmin
            distances_matrix_list.append(distances_matrix)
            #if ID == 5:
            #    return distances_matrix_list
        return distances_matrix_list


def precompute_all_optimized(Context, Observation):
        Tb=Observation['Tb']
        K=Observation['Wt']
        VType=Observation['VType']
        #n_ells=K.shape[0]
        n_Vars=K.shape[1]
        n_Ells=K.shape[0]
        
        SolH=np.zeros(n_Vars)
        SolV=np.zeros(n_Vars)
        KGrid=Observation['KGrid']
        CIdG=Context['Dict_Grids']['CellID_Grid']
        LTG=Context['Dict_Grids']['LandType_Grid']##
        X=range(CIdG.shape[0])##
        Y=range(CIdG.shape[1])##
        #M=np.mgrid[0:CIdG.shape[0],0:CIdG.shape[1]]##
        #A=np.zeros([n_Vars,n_Ells])

        print('Precomputing distances for Backus Gilbert (fine, %d cells)'%n_Vars)
        #print("gamma: %.3f, w: %.5f, error_std: %.3f "%(gamma,w,error_std))
        distances_matrix_list = []
        J=CIdG*0
        nx,ny=J.shape
        for ID in range(n_Vars):
            print("Preprocesando ", ID, " de ", n_Vars)
            distances_matrix = np.zeros(J.shape)
            [Wx,Wy]=np.where(CIdG==ID)
            mx=Wx.min()
            Mx=Wx.max()
            my=Wy.min()
            My=Wy.max()
            cx=(mx+Mx)/2
            cy=(my+My)/2
            width=Mx-mx
            height=My-my
            area_celda = len(Wx)
            area_bbox = (width+1)*(height+1)            
            #print("Area de celda: ", area_celda, " Area de bbox: ", area_bbox)            
            if area_celda != area_bbox:
                # Es una celda irregular...
                print("Celda irregular, busqueda exhaustiva")                
                print("Area de celda: ", area_celda, " Area de bbox: ", area_bbox)
                for i in range(nx):
                    print("Elemento i: ", i)                
                    for j in range(ny):
                        dmin = None
                        for k in range(len(Wx)):
                            puntoX = Wx[k]
                            puntoY = Wy[k]
                            d = (i - puntoX)**2 + (j-puntoY)**2
                            if dmin == None or d < dmin:
                                dmin = d
                        #distances_matrix[i,j]=np.sqrt(dx**2 + dy**2)
                        distances_matrix[i,j]=dmin
            else:
                print("Celda regular")
                for i in range(nx):
                    for j in range(ny):
                        dx = np.max((0.0, np.abs(i-cx) - width/2.0))                    
                        dy = np.max((0.0, np.abs(j-cy) - height/2.0))                    
                        #distances_matrix[i,j]=np.sqrt(dx**2 + dy**2)
                        distances_matrix[i,j]=dx**2 + dy**2

            distances_matrix_list.append(distances_matrix)
            #if ID == 5:
            #    return distances_matrix_list
        return distances_matrix_list




#distances_matrix_list = precomputeBBox(Context, Observation)

#plt.imshow(distances_matrix_list[2])
#plt.show()

#distances_matrix_list = precompute_all(Context, Observation)

distances_matrix_list = precompute_all_optimized(Context, Observation)

# Pickleado.

# Guardar
with open('/home/rgrimson/data_10km.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(distances_matrix_list, f, pickle.HIGHEST_PROTOCOL)

# Abrir
with open('data_25km.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    distances_matrix_list_recovered = pickle.load(f)

with open('/home/rgrimson/data_10km.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    distances_matrix_list_recovered = pickle.load(f)



BG_Coef=L_Solvers.BG_Precomp_MaxObs_V3_c(distances_matrix_list_recovered, Context, Observation, gamma=np.pi/4,w=0.001,MaxObs=4)
Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
RMSE=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
print("RMSE: %.5f\n*****************************************"%RMSE)        


# Ver si no es J al cuadrado.


#Sol,RMSE = BG(V=3)
sys.exit(1)    
#%%    
GAMMAS=np.linspace(0,np.pi/2,20)    


RMSE1_=[]
RMSE2_=[]
RMSE4_=[]
#%%
RMSE3 = []
for g in GAMMAS:
    BG_Coef=L_Solvers.BG_Precomp_MaxObs_V3_c(distances_matrix_list, Context, Observation, gamma=g,w=0.0001,MaxObs=4)
    Sol=[BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
    RMSE=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
    print("RMSE: %.5f\n*****************************************"%RMSE)        
    RMSE3.append(RMSE)

plt.plot(GAMMAS, RMSE3)
#%%
for g in GAMMAS:
    Sol, RMSE1 = BG(V=1,gamma=g)
    RMSE1_.append(RMSE1)
    Sol, RMSE2 = BG(V=2,gamma=g,w=0.1)
    RMSE2_.append(RMSE2)
    Sol, RMSE4 = BG(V=4,gamma=g)
    RMSE4_.append(RMSE4)

Sol = L_Solvers.Solve_Rodgers_IT_NB(Context,Observation, tita,)
RMSE=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
print("RMSE: %.5f\n*****************************************"%RMSE)        


