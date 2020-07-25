# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:48:27 2018

@author: Juan Charles
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:47:57 2017

@author: rgrimson
"""
from __future__ import print_function


#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
MAPA='Ajo_10km'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA


import sys

import L_ReadObs
import L_Context
import L_Kernels as L_Kernels
import L_Output
import L_Solvers

import copy

import L_ParamEst
import L_Syn_Img
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

#%%

# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

#Bands=['ku','ka','c','x','K']
Bands=['X']#,'K','KU','C','X']
#Bands =  ['KA','K']
NeighborsBehaviour = L_Context.LOAD_NEIGHBORS
Methods = ['LSQR']#,'Weights','Rodgers_IT',"Global_GCV_Tichonov",'BGF']
#Methods = ['Rodgers_IT']
Img_rep=4
 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS

#COORDENADAS A SHIFTEAR, OBSERVAR 0,0 ES LA ORIGINAL
x_shift = [-5000,-4000,-3000,-2000,-1000,0,1000,2000,3000,4000,5000]
y_shift = [0]

dic_maxObs={'KA10':6,'KU10':6,'K10':6,'X10':4,'C10':4,'KA25':20,'KU25':20,'K25':20,'X25':12,'C25':12}

df_summary= pd.read_csv('/home/mariela/Desktop/exp_review/PMW/summary_gamma_values.csv')
df_summary=df_summary[['Pol', 'MaxObs', 'km', 'Band', 'Gamma']]


csv_name = 'shift_soloX_10km_conRMSE_border4'
#csv_name = 'exp_bands_methods_shifts_25km'
#%%
import itertools

columnas = ['shift_x','shift_y','Method','Band','Pol','Img_num','Mu_real_0','Sigma2_real_0','Mu_real_1','Sigma2_real_1','Mu_0','Sigma2_0','Mu_1','Sigma2_1','RSME','RMSE_border','CoefCorr','time'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)

for Band in Bands:
    for shift in list(itertools.product(x_shift,y_shift)):
        #Load Context    
        Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
        #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
        L_Syn_Img.Set_Margins(Context)
        #Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
        Observation, tita=L_ReadObs.Load_Shifted_Band_Kernels(HDFfilename,Context,Band,shift[0],shift[1])
    
                
        TITA_Synth={}    
        TITA={}
        RMSE={}
        COEF_CORR = {}
        Obs_error_std = 1

        for rep in range(Img_rep): # En cada repeticion arma una imagen sintetica distinta para cada banda, esta sería l nueva verdad del terreno.

            
            TbH=L_Syn_Img.Create_Random_Synth_Img(Context)
            TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
            
            
            #Export Synthetic Image Map
            #L_Output.Export_Solution([TbH,TbV], Context, "Synth", "SynthImg_r"+str(rep))

            TITA_Synth[rep]={}
            RMSE[rep]={}
            TITA[rep]={}
            COEF_CORR[rep] = {}
            
    
            #obs_std_errors += str(Obs_error_std) + " "
    
            Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_error_std)
            TITA_Synth[rep]={}
            TITA[rep]={}
            RMSE[rep]={}
            COEF_CORR[rep] = {}
            tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
            TITA_Synth[rep]=copy.deepcopy(tita)

            tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)

            for Method in Methods:
                tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
                t_inicial = time.time()
                if Method == 'BGF':
                    gamma_h=df_summary[(df_summary['Pol']=='H')&(df_summary['km']==int(MAPA[4:6]))&(df_summary['Band']==Band)]['Gamma'].iloc[0]
                    gamma_v=df_summary[(df_summary['Pol']=='V')&(df_summary['km']==int(MAPA[4:6]))&(df_summary['Band']==Band)]['Gamma'].iloc[0]
                
                    BG_Coef_H=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma_h,w=0.001,MaxObs=dic_maxObs[Band+MAPA[4:6]])
                    BG_Coef_V=L_Solvers.BG_Precomp_MaxObs(Context, Observation, gamma=gamma_v,w=0.001,MaxObs=dic_maxObs[Band+MAPA[4:6]])
                    Sol=[BG_Coef_H.dot(Observation['Tb']['H']), BG_Coef_V.dot(Observation['Tb']['V'])]
                else:
                    Sol=L_Solvers.Solve(Context,Observation,Method, tita)


                t_final = time.time()
                t = t_final - t_inicial

 
                titaReal=copy.deepcopy(L_ParamEst.recompute_param([TbH,TbV],Observation,tita))
                L_ParamEst.pr(titaReal,'Synthetic Image')

                tita=L_ParamEst.recompute_param(Sol,Observation,tita)

                L_ParamEst.pr(tita, 'Reconstructed Image')


                RMSE[rep][Method]={}
                RMSE[rep][Method]['H']=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                RMSE[rep][Method]['V']=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
                COEF_CORR[rep][Method] = {}
                COEF_CORR[rep][Method]['H'] = Coef_correlacion(TbH,Sol[0])
                COEF_CORR[rep][Method]['V'] = Coef_correlacion(TbV,Sol[1])

                TITA[rep][Method]=copy.deepcopy(tita)
                RMSE_border=  L_Syn_Img.RMSE_M_Border(TbH,TbV,Sol, Context,Band)
    
                dic = {'shift_x':shift[0],'shift_y':shift[1],'Method': Method,'Band':Band,'Pol': 'H','Img_num': rep,'Mu_real_0':titaReal['H']['mu'][0],'Sigma2_real_0':titaReal['H']['sigma2'][0],'Mu_real_1':titaReal['H']['mu'][1],'Sigma2_real_1':titaReal['H']['sigma2'][1],'Mu_0':tita['H']['mu'][0],'Sigma2_0':tita['H']['sigma2'][0],'Mu_1':tita['H']['mu'][1],'Sigma2_1':tita['H']['sigma2'][1],'RSME':str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)),'RMSE_border':RMSE_border[0],'CoefCorr':COEF_CORR[rep][Method]['H'],'time':t}
                df = df.append(dic, ignore_index=True)
                dic = {'shift_x':shift[0],'shift_y':shift[1],'Method': Method,'Band':Band,'Pol': 'V','Img_num': rep,'Mu_real_0':titaReal['V']['mu'][0],'Sigma2_real_0':titaReal['V']['sigma2'][0],'Mu_real_1':titaReal['V']['mu'][1],'Sigma2_real_1':titaReal['V']['sigma2'][1],'Mu_0':tita['V']['mu'][0],'Sigma2_0':tita['V']['sigma2'][0],'Mu_1':tita['V']['mu'][1],'Sigma2_1':tita['V']['sigma2'][1],'RSME':str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)),'RMSE_border':RMSE_border[1],'CoefCorr':COEF_CORR[rep][Method]['V'],'time':t}
                df = df.append(dic, ignore_index=True)
    
    
                print('termino:')
                print(Band)
                print(Method)
                print(rep)
                print(shift)


df.to_csv(wdir + '/Salidas/%s.csv'%csv_name)

#%%

#df = pd.read_csv(wdir + '/Salidas/%s.csv'%csv_name)

#Method_sel='Rodgers_IT'
#Method_sel = 'Rodgers_IT'#], 'LSQR', 'Weights', 'Global_GCV_Tichonov']
#Band ='KU'
#df_method = df[(df['Method']==Method_sel)&(df['Band']==Band)]

#df_method_group = df_method.groupby([ 'Band', 'Pol','shift_x', 'shift_y']).mean()

#df_method_group = df_method_group[['RSME', 'CoefCorr']]

#df_method_group.to_csv(wdir + '/Salidas/shift_'+Method_sel+'_'+Band+'25km.csv')

#%%

#%%

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

csv_name = 'shift_10km_conRMSE_border_vf'

df = pd.read_csv(wdir + 'Salidas/%s.csv'%csv_name)
Method_sel ='Global_GCV_Tichonov'
Method_sel = 'LSQR'
Band = 'X'
Pol ='V'

Bands=['KA','K','KU','C','X']

Methods = ['Rodgers_IT',"Global_GCV_Tichonov"] 
for Band in Bands:
    for Method in Methods:
        plt.clf()
        df1 = df[(df['Method']==Method)&(df['Band']==Band)&(df['Pol']==Pol)]
        
        #df1 = df1[['shift_x', 'shift_y','Img_num','RSME', 'CoefCorr']]
        
        df1 = df1[['shift_x', 'shift_y','Img_num','RSME', 'CoefCorr']]
                
        
        plt.scatter(df1['shift_x'],df1['RSME'])
        plt.ylabel('RSME')
        plt.xlabel('shift_x')
        plt.title(Band+'  '+Pol+'  '+Method)
        
        df1 = df1.groupby(['shift_x', 'shift_y']).mean()
        df1 = df1.reset_index()
        
        plt.plot(df1['shift_x'],df1['RSME'])
        #plt.ylabel('RSME mean')
        plt.xlabel('shift_x')
        plt.title(Band+'  '+Pol+'  '+Method)
        plt.savefig(wdir + '/Salidas/shift_25km_'+Method+'_'+Band+'_'+Pol+'.jpg', bbox_inches='tight')