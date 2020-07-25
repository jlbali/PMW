# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:15:19 2017

@author: mariela
"""

# ANTES DE COMENZAR REVISAR MAPA Y EL NOMBRE DEL CSV

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


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

# INICIALIZO LOS VALORES A EXPERIMENTAR:
#%%
gammas =np.linspace(0,np.pi/2,100)

ws = [0.001]#,0.10]#np.linspace(0.0, 1000.0, num=11).tolist()


HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'

NeighborsBehaviour = L_Context.LOAD_NEIGHBORS
Method = 'BGF'
#Bands=['KA','K','KU','C','X']
Bands=['X']

Img_rep=1 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS

csv_name = 'x_w001_gammas_25km_100gammas'
#csv_name ='exBGF_GammaW_25_ka_step'
# ESTA FIJADO CON RUIDO CUYA VARIANZA 1.

#%%

columnas = ['Method','Band','Pol','Img_num','gamma','w','Mu_real_0','Sigma2_real_0','Mu_real_1','Sigma2_real_1','Mu_0','Sigma2_0','Mu_1','Sigma2_1','RSME','CoefCorr','time'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)



for Band in Bands:
    
    #Load Context    
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
    L_Syn_Img.Set_Margins(Context)
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=False)

    TITA_Synth={}    
    TITA={}
    RMSE={}
    COEF_CORR = {}
    Obs_error_std = 1
    
    # Genero el df al que le voy a ir appendeando la info

    

    for rep in range(Img_rep): # En cada repeticion arma una imagen sintetica distinta para cada banda, esta sería l nueva verdad del terreno.
        
        #Create Synthetic Image

        
        
        TbH=L_Syn_Img.Create_Random_Synth_Img(Context)
        #TbV=L_Syn_Img.Create_Random_Synth_Img(Context)
        #TbH = L_Syn_Img.Create_Trig_Synth_Img(Context)
        TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
        
        
        #Export Synthetic Image Map
        #L_Output.Export_Solution([TbH,TbV], Context, "Synth", "SynthImg_r"+str(rep))
        
        # #%%
        s = ""
        #Load ellipses
        #Set Synthetic Obsetvations
        
        #Genero Obs_error_std, un opcion seria poner 
        TITA_Synth[rep]={}
        RMSE[rep]={}
        TITA[rep]={}
        COEF_CORR[rep] = {}
        

        Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=1)
        TITA_Synth[rep]={}
        TITA[rep]={}
        RMSE[rep]={}
        COEF_CORR[rep] = {}
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        TITA_Synth[rep]=copy.deepcopy(tita)
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        s+= L_ParamEst.toString(tita, tita, 'Synthetic Image')
#   
   
            

        print("\n·································································")            
        print("........... SOLVING BAND %s with Method %s" %(Band, Method))
        s += "............. SOLVING BAND: " + str(Band) + "\n"
        s += "Method: " + str(Method) + "\n"
        s += ".............................................\n"
        for gamma in gammas:
                    for w in ws:
                        
    #    
    #                #%##################################################
    #                ## SOLVE SYSTEM with synth sample parameters as input
                        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
                        t_inicial = time.time()
                        Sol=L_Solvers.BGF(Context,Observation,gamma,w)
                        t_final = time.time()
                        t = t_final - t_inicial
            #            
            #                #%##################################################
            #                ## Export solution
                        #L_Output.Export_Solution(Sol, Context, Band, "SolSynthImg_%s_r%d_"%(Method,rep))
            #                #%    
            #                #Compare Original and Reconstructed: Evaluate error (with and without margins)
                        print("\nSummary:\n--------")
                        print("\nSummary:\n--------")
                        print("RMSE H: %.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)))
                        print("RMSE V: %.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
                        for lt in range(2):
                            print("    LT: %d"%(lt))
                            print("RMSE H: %.3f"%(L_Syn_Img.RMSE_M_LT(TbH,Sol[0],Context, lt)))
                            print("RMSE V: %.3f"%(L_Syn_Img.RMSE_M_LT(TbV,Sol[1],Context, lt)))    
                        titaReal=copy.deepcopy(L_ParamEst.recompute_param([TbH,TbV],Observation,tita))
                        L_ParamEst.pr(titaReal,'Synthetic Image')
            #                
                        tita=L_ParamEst.recompute_param(Sol,Observation,tita)
                        s += L_ParamEst.toString(tita, titaReal, 'Reconstructed Image_'+str(rep))
                        L_ParamEst.pr(tita, 'Reconstructed Image')
            
                        
                        
                        
                        print("\nErrors:\n--------")
                        print("RMSE H: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context), L_Syn_Img.RMSE(TbH,Sol[0]))) # ¿Tiene sentido calcularlo en cada iteracion?
                        print("RMSE V: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context), L_Syn_Img.RMSE(TbV,Sol[1])))
                        s += "RMSE H: " + str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)) + "\n"
                        s += "RMSE V: " + str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)) + "\n"
                        RMSE[rep][Method]={}
                        RMSE[rep][Method]['H']=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                        RMSE[rep][Method]['V']=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
                        COEF_CORR[rep][Method] = {}
                        COEF_CORR[rep][Method]['H'] = Coef_correlacion(TbH,Sol[0])
                        COEF_CORR[rep][Method]['V'] = Coef_correlacion(TbV,Sol[1])
                        print("\nErrors:\n--------")
                        print("CoeficienteCorrelacion H: %.3f"%(COEF_CORR[rep][Method]['H'])) # ¿Tiene sentido calcularlo en cada iteracion?
                        print("CoeficienteCorrelacion V: %.3f"%( COEF_CORR[rep][Method]['V']))
                        s += "CoeficienteCorrelacion H: " + str(( COEF_CORR[rep][Method]['H'])) + "\n"
                        s += "CoeficienteCorrelacion V: " + str(( COEF_CORR[rep][Method]['V'])) + "\n"
                        TITA[rep][Method]=copy.deepcopy(tita)
            
                        dic = {'Method': Method,'Band':Band,'Pol': 'H','Img_num': rep,'gamma':gamma,'w':w,'Mu_real_0':titaReal['H']['mu'][0],'Sigma2_real_0':titaReal['H']['sigma2'][0],'Mu_real_1':titaReal['H']['mu'][1],'Sigma2_real_1':titaReal['H']['sigma2'][1],'Mu_0':tita['H']['mu'][0],'Sigma2_0':tita['H']['sigma2'][0],'Mu_1':tita['H']['mu'][1],'Sigma2_1':tita['H']['sigma2'][1],'RSME':str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)),'CoefCorr':COEF_CORR[rep][Method]['H'],'time':t}
                        df = df.append(dic, ignore_index=True)
                        dic = {'Method': Method,'Band':Band,'Pol': 'V','Img_num': rep,'gamma':gamma,'w':w,'Mu_real_0':titaReal['V']['mu'][0],'Sigma2_real_0':titaReal['V']['sigma2'][0],'Mu_real_1':titaReal['V']['mu'][1],'Sigma2_real_1':titaReal['V']['sigma2'][1],'Mu_0':tita['V']['mu'][0],'Sigma2_0':tita['V']['sigma2'][0],'Mu_1':tita['V']['mu'][1],'Sigma2_1':tita['V']['sigma2'][1],'RSME':str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)),'CoefCorr':COEF_CORR[rep][Method]['V'],'time':t}
                        df = df.append(dic, ignore_index=True)
            
            
                        print('termino:')
                        print(Band)
                        print(Method)
                        print(rep)
                        print(gamma)
                        print(w)
        

print("RESULTADOS FINALES \n\n\n")
s += "El archivo HDF es:" + str(HDFfilename)
print(s)

df.to_csv(wdir + '/Salidas/%s.csv'%csv_name)

#%%


#%%
df = pd.read_csv(wdir + '/Salidas/%s.csv'%csv_name)

df_group_mean = df.groupby(['Pol','gamma','w']).mean()
df_group_std = df.groupby(['Pol','gamma','w']).std()

#%%
df_group_m = df_group_mean[['RSME', 'CoefCorr']]
df_group_m =df_group_m.reset_index()

df_group_s = df_group_std[['RSME','CoefCorr']]  
df_group_s =df_group_s.reset_index()
#%%

result = pd.merge(df_group_m, df_group_s,on=['Pol','gamma','w'])
result.columns = [['Pol', 'gamma', 'w', 'RSME_mean', 'CoefCorr_mean', 'RSME_std', 'CoefCorr_std']]


result_H = result[result['Pol']=='H']
plt.scatter(result_H['gamma'],result_H['RSME_mean'],color='r')
#plt.legend()
plt.xlabel('gamma')
plt.ylabel('RMSE mean')

result_V = result[result['Pol']=='V']
plt.scatter(result_V['gamma'],result_V['RSME_mean'],color='g')
#plt.legend()
plt.xlabel('gamma')
plt.ylabel('RMSE mean')