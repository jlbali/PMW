# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:03:26 2017

@author: mariela
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

#from __future__ import print_function

#WDIR
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
import matplotlib.pyplot as plt
import pandas as pd
import time


def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]


#%%
##PROCESS PARAMETERS
#HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
##Bands=['ka','K','ku','x','c']
#Band='ka'
#Obs_error_std=0.5
#NeighborsBehaviour = L_Context.LOAD_NEIGHBORS
#Methods = ["Weights", "Global_GCV_Tichonov", "K_Global_GCV_Tichonov","Rodgers_IT",'LSTSQR','BGF']
#N_rep=3#3



#%%

#Primer ejemplo
#PASADA 1
HDFfilename='GW1AM2_201208110420_224D_L1SGBTBR_2210210'
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'


# PASADA 2
#HDFfilename='GW1AM2_201208011744_113A_L1SGBTBR_2210210'


#PASADA3
#HDFfilename = 'GW1AM2_201207310439_227D_L1SGBTBR_2210210'

Bands=['KA','KU','K','X','C']
#Band='ka'
#Obs_error_std=0.5
NeighborsBehaviour = L_Context.LOAD_NEIGHBORS
Methods = ['LSQR','Weights','Rodgers_IT']
#Methods = ['LSQR','Weights','Rodgers_IT',"Global_GCV_Tichonov",'K_Global_GCV_Tichonov']

N_rep=100

#'Naive_Bayes','Naive_Bayes_IT','LSQR','Weights','Rodgers','RodgersFromElls','Rodgers_IT','LSTSQR','Tichonov','GCV_Tichonov',"Global_GCV_Tichonov","K_GCV_Tichonov","K_Global_GCV_Tichonov"


#%%

def main(): 
    #%%###################
    ## PREPROCESSING    
    l = []
    for Band in Bands:
        #Load Context    
        Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
        #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
        L_Syn_Img.Set_Margins(Context)
        Observation, tita=L_ReadObs.Load_Band_Kernels(HDFfilename, Context, Band)#Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)
    
        TITA_Synth={}    
        TITA={}
        RMSE={}
        COEF_CORR = {}
        Obs_error_std = 1
        l_i = []
    
        columnas = ['Metodo','Banda','Repeticion','i','time_lapsed*2','Mu lt=0 Real','Mu H lt=0','Sigma2 H lt=0 Real','Sigma2 H lt=0','Mu H lt=1 Real','Mu lt=1 H','Sigma2 H lt=1 Real','Sigma2 H lt=1','Mu V lt=0 Real','Mu V lt=0','Sigma2 V lt=0 Real','Sigma2 V lt=0','Mu lt=1 V Real','Mu V  lt=1','Sigma2 V lt=1 Real','Sigma2 V lt=1','RMSE H','RMSE V','CoeficienteCorrelacion H','CoeficienteCorrelacion V']
    
        for rep in range(N_rep): # En cada repeticion arma una imagen sintetica distinta para cada banda, esta sería l nueva verdad del terreno.
            
            #Create Synthetic Image
    
            
            
            TbH=L_Syn_Img.Create_Random_Synth_Img(Context,sgm=5)
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
            
            for i in range(0,1):
            #obs_std_errors += str(Obs_error_std) + " "
                
                Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_error_std)
                TITA_Synth[rep][i]={}
                TITA[rep][i]={}
                RMSE[rep][i]={}
                COEF_CORR[rep][i] = {}
                tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
                TITA_Synth[rep][i]=copy.deepcopy(tita)
    #            
                tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
                s+= L_ParamEst.toString(tita, tita, 'Synthetic Image')
    #   
    #            
                for Method in Methods:
               
                    
                    l_i.append(Method)
                    l_i.append(Band)
                    l_i.append(rep)
                    l_i.append(i)
                    print("\n·································································")            
                    print("........... SOLVING BAND %s with Method %s" %(Band, Method))
                    s += "............. SOLVING BAND: " + str(Band) + "\n"
                    s += "Method: " + str(Method) + "\n"
                    s += ".............................................\n"
    #    
    #                #%##################################################
    #                ## SOLVE SYSTEM with synth sample parameters as input
                    tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
                    t_inicial = time.time()
                    Sol=L_Solvers.Solve(Context,Observation,Method, tita)
                    t_final = time.time()
                    t = t_final - t_inicial
    #            
    #                #%##################################################
    #                ## Export solution
                    #L_Output.Export_Solution(Sol, Context, Band, "SolSynthImg_%s_r%d_%i"%(Method,rep,i))
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

                    l_i.append(t)                    
                    
                    l_i.append(titaReal['H']['mu'][0])
                    l_i.append(tita['H']['mu'][0])

                    l_i.append(titaReal['H']['sigma2'][0])
                    l_i.append(tita['H']['sigma2'][0])
                                    
                    l_i.append(titaReal['H']['mu'][1])
                    l_i.append(tita['H']['mu'][1])
                    
                    l_i.append(titaReal['H']['sigma2'][1])
                    l_i.append(tita['H']['sigma2'][1])
                                    
                                    
                    l_i.append(titaReal['V']['mu'][0])
                    l_i.append(tita['V']['mu'][0])

                    l_i.append(titaReal['V']['sigma2'][0])
                    l_i.append(tita['V']['sigma2'][0])
                                    
                    l_i.append(titaReal['V']['mu'][1])
                    l_i.append(tita['V']['mu'][1])
                    
                    l_i.append(titaReal['V']['sigma2'][1])
                    l_i.append(tita['V']['sigma2'][1])
                    
                    print("\nErrors:\n--------")
                    print("RMSE H: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context), L_Syn_Img.RMSE(TbH,Sol[0]))) # ¿Tiene sentido calcularlo en cada iteracion?
                    print("RMSE V: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context), L_Syn_Img.RMSE(TbV,Sol[1])))
                    s += "RMSE H: " + str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)) + "\n"
                    s += "RMSE V: " + str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)) + "\n"
                    RMSE[rep][i][Method]={}
                    RMSE[rep][i][Method]['H']=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
                    RMSE[rep][i][Method]['V']=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
                    COEF_CORR[rep][i][Method] = {}
                    COEF_CORR[rep][i][Method]['H'] = Coef_correlacion(TbH,Sol[0])
                    COEF_CORR[rep][i][Method]['V'] = Coef_correlacion(TbV,Sol[1])
                    print("\nErrors:\n--------")
                    print("CoeficienteCorrelacion H: %.3f"%(COEF_CORR[rep][i][Method]['H'])) # ¿Tiene sentido calcularlo en cada iteracion?
                    print("CoeficienteCorrelacion V: %.3f"%( COEF_CORR[rep][i][Method]['V']))
                    s += "CoeficienteCorrelacion H: " + str(( COEF_CORR[rep][i][Method]['H'])) + "\n"
                    s += "CoeficienteCorrelacion V: " + str(( COEF_CORR[rep][i][Method]['V'])) + "\n"
                    TITA[rep][i][Method]=copy.deepcopy(tita)
                    l_i.append(str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)))
                    l_i.append(str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)))
                    l_i.append(COEF_CORR[rep][i][Method]['H'])
                    l_i.append(COEF_CORR[rep][i][Method]['V'])
                    l.append(l_i)
                    l_i=[]
                    print('termino:')
                    print(Band)
                    print(Method)
                    print(i)
                    print(rep)


    print("RESULTADOS FINALES \n\n\n")
    s += "El archivo HDF es:" + str(HDFfilename)
    print(s)
    df = pd.DataFrame(l,columns = columnas)
    df.to_csv(wdir + 'Salidas/test_25km_100imagenes_mu150_280_std_allbands_v6_sigma510_k020.csv')
    print("Generando archivo...")
    f = open(wdir + 'Salidas/test_25km_100imagenes_mumu150_280_std_allbands_v6_sigma510_k020.txt', 'w')
    f.write(s) # Ahora si anda, no le gustaba el nombre solo del archivo, por algun motivo...
    f.close()
#%%            

        

if __name__ == "__main__":
    main()

