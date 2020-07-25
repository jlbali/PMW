# -*- coding: utf-8 -*-

Band = 'X'
tot_obs_samples=1 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG

BASE_DIR="/home/rgrimson/Projects/PMW_Tychonov/"
csv_dir=BASE_DIR + 'Output/'

#Import libraries
import copy
import L_ReadObs
import L_Context
import L_Output
import L_NewSolvers
import L_ParamEst
import L_Syn_Img
import numpy as np
import pandas as pd
import time
import pickle
import sys
import matplotlib.pyplot as plt


# Tichonov no requiere saber calibracion instrumental...

#%%
# INPUTS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
#csv_name ='simple_Trigonometric_2'
csv_name ='exp3_' + Band

Bands = [Band]


#MAPAS=['LCA_25000m'] # Solo para trigonometric.
MAPAS=['Reg25']


base_type = "SLOPE" # Usar mapas LCA para esto.


Methods = ['EM','GCV_Tichonov','LSQR']

Obs_errors = [1.0]

np.random.seed(1)
columnas = ['CellSize','Method','Band','mix_ratio','Img_num','Obs_num','Obs_std_error','RMSE','CoefCorr','time'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)

mix_ratios = [1.0]

n_img = 1


export_real = True
export_solutions = True
show_histograms = True

for MAPA in MAPAS:
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    print("Contexto cargado en", wdir)
    #print("Context", Context)
    #print("Keys: ", Context.keys())
    #print("sdfsdf: ", Context['Dict_Vars'].keys())
    VType = Context['Dict_Vars']["VType"]
    neighbors = Context["Neighbors"]
    extras ={ 
        "neighbors": neighbors,
        "obsstd": 1.0,        
    }

    L_Syn_Img.Set_Margins(Context)
    print("Margenes setados para mapa", MAPA)
    km=int(MAPA[4:6])
    Observations = {}
    titas = {}
    v_errs = {}
    for Band in Bands:
        Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
        Observations[Band] = Observation
        titas[Band] = tita
        K=Observation['Wt']
        n_cells = K.shape[1] # No cambia con la banda.
        n_ell = K.shape[0]
        v_errs[Band] = {}
        for Obs_std in Obs_errors:
            v_errs[Band][Obs_std] = []
            for n_obs in range(tot_obs_samples):
                v_err=np.random.normal(0,Obs_std,n_ell)
                v_errs[Band][Obs_std].append(v_err)


    if base_type == "BIMODAL":
        #Tb_base = L_Syn_Img.Create_Bimodal_Synth_Img(Context, mu=180, sdev=0.3)
        Tb_base = L_Syn_Img.Create_True_Bimodal_Synth_Img(Context)
        Tb_random = 230 + np.random.normal(0,10,n_cells)
        L_Output.Export_Solution([Tb_base,Tb_base], Context, "RealBase", 'RealBase' )                                 
    elif base_type =="TRIGONOMETRIC_ONE_TYPE":
        Tb_base = L_Syn_Img.Create_Trig_Synth_Img0(Context)
        Tb_random = L_Syn_Img.Create_Random_Synth_Img(Context)
    elif base_type == "TRIGONOMETRIC_TWO_TYPES":
        Tb_base = L_Syn_Img.Create_Trig_Synth_Img(Context)
        Tb_random = L_Syn_Img.Create_Random_Synth_Img(Context)
    elif base_type == "SLOPE":
        print("Slope method")
        Tb_base = L_Syn_Img.Create_Slope_Synth_Img(Context)
        Tb_random = L_Syn_Img.Create_Random_Synth_Img(Context)
        print("Fin Slope Method")
    else:
        print("Desconocido")
        sys.exit(0)
        
        
    for mix_ratio in mix_ratios:
        """            
        if base_type == "BIMODAL":
             Tb = Tb_base + (1-mix_ratio)*Tb_random
        else:
             Tb = mix_ratio*Tb_base + (1-mix_ratio)*Tb_random
        """
        Tb = mix_ratio*Tb_base + (1-mix_ratio)*Tb_random
        if export_real:
             Sol = [Tb, Tb]
             L_Output.Export_Solution(Sol, Context, "Slope", 'Slope_'+str(km)+'_' +str(mix_ratio).replace('.','_') )                     
    

        for Band in Bands:
            Observation = Observations[Band]
            tita = titas[Band]
            print("Banda", Band, "cargada")
            K=Observation['Wt']
            n_ell = K.shape[0]
            n_cells = K.shape[1]
        
            for Obs_std in Obs_errors:
               titaReal=copy.deepcopy(L_ParamEst.Compute_NoPol_param(Tb,VType))
               for n_obs in range(tot_obs_samples):
                    v_err=v_errs[Band][Obs_std][n_obs]
                    Tb_sim=L_Syn_Img.Simulate_NoPol_PMW(K,Tb,Obs_error_std=0)
                    Tb_sim+=v_err
                    for Method in Methods:
    
    
                        t_inicial = time.time()                
                        #corrergir esto!!!

                        Sol=L_NewSolvers.Solve(VType, K, Tb_sim, Method, extras)
            
                        t_final = time.time()
                        t = t_final - t_inicial
                        
                        L_ParamEst.pr_NoPol(titaReal,'Synthetic Image')
                        tita=L_ParamEst.Compute_NoPol_param(Sol, VType)
                        L_ParamEst.pr_NoPol(tita, 'Reconstructed Image')
                
                        dic = {'CellSize':km,'Method': Method,'Band':Band,'Img_num': n_img,'Obs_num': n_obs,'Obs_std_error':Obs_std, 'mix_ratio': mix_ratio,
                          'RMSE':(L_Syn_Img.COMP_COND(Tb,Sol,Context,Margin=False)[0]),
                          'CoefCorr':(L_Syn_Img.COMP_COND(Tb,Sol,Context,Margin=False)[1]),
                          'time':t}
                        df = df.append(dic, ignore_index=True)
        
                        print('termino:',km,Band,Method,Obs_std,n_img,n_obs)
                        if export_solutions:
                            Sol2 = [Sol, Sol]
                            L_Output.Export_Solution(Sol2, Context, Band, 'Slope_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_') + "_"  +str(mix_ratio).replace('.','_') )
                        if show_histograms:
                            plt.hist(Sol)
                            plt.show()
                        
                        

df.to_csv(csv_dir + csv_name + '.csv')

sys.exit(0)            


