
# EXPERIMENTO 1

Band = 'KU'
tot_img_samples=3 #CANTIDAD DE IMAGENES SINTETICAS GENERADAS
tot_obs_samples=3 #CANTIDAD DE OBS SINTETICAS GENERADAS por IMG







# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 09:45:54 2018

@author: mariela
"""

# ADAPTAR ESTO PARA EL NUEVO ESQUEMA SIN POLARIZACION.
"""
Problemas:
    - GCV Tichonov aparente devolver el lambda mas bajo posible...
    - Siempre hace el geocorrecting, y no guarda nada en el GC.

Para que librería hdf5 ande en el Anaconda es preciso tener la última versión.

ESTE ESTA ANDANDO MEDIO RARO
"""

#import my_base_dir
#BASE_DIR = my_base_dir.get()
BASE_DIR="/home/lbali/PMW-Tychonov/"
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
csv_name ='exp1_' + Band

Bands = [Band]

#%%

MAPAS=['LCA_12500m','LCA_25000m','LCA_50000m']
MAPAS=['LCA_50000m','LCA_25000m','LCA_12500m']
MAPAS=['LCA_25000m'] # Solo para trigonometric.
#MAPAS=['Reg25'] # Solo para bimodal.


#Methods = ['LSQR','Weights','EM','Rodgers_IT','Global_GCV_Tichonov','BGF']
#Methods = ['EM','BGF','LSQR']
Methods = ['EM','GCV_Tichonov','LSQR']


#Bands=['KA','KU','K','X','C']
#Bands=['KA']
#Bands=[ 'X'] # Probar cambiando la banda.
#Bands=[ 'KU']
#Bands=['KA','KU','K','X','C']
#Bands=['K']
#base_type = "TRIGONOMETRIC_ONE_TYPE"
base_type = "TRIGONOMETRIC_TWO_TYPES" # Usar mapas LCA para esto.
#base_type = "BIMODAL"  # Usar mapas Reg para esto.
#tipo ="RANDOM"
#tipo = "MIXED"
#mix_ratios = [1.0]
mix_ratios = np.linspace(0.0, 1.0, 11)
#mix_ratios = [0.0, 1.0]



#Obs_errors = [0.25, 0.5,1.0, 2.0,4.0]
#Obs_errors = [1.0]
#Obs_errors = [2.0]
Obs_errors = [1.0]
# Para probar con distintos obs_errors... por màs y por menos.

export_solutions = False
export_real = False
#%%

# Levantamos el diccionario de coeficientes de Backus-Gilbert..
dic_BG_coef = {}
if 'BGF' in Methods:
    for MAPA in MAPAS:
        dic_mapa = {}
        dic_BG_coef[MAPA] = dic_mapa
        for Band in Bands:
            filepath = csv_dir + "BG_" + MAPA + "_" + Band + ".pickle"
            with open(filepath, 'rb') as f:
                # The protocol version used is detected automatically, so we do not
                # have to specify it.
                data = pickle.load(f)
                dic_mapa[Band] = data



np.random.seed(1)
columnas = ['CellSize','Method','Band','mix_ratio','Img_num','Obs_num','Obs_std_error','RMSE','CoefCorr','time'] #el 0 y el 1 correspondonen al LT
df = pd.DataFrame(columns=columnas)



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

    for n_img in range(tot_img_samples):
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
        else:
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
                 L_Output.Export_Solution(Sol, Context, "Real", 'Real_'+str(km)+'_' +str(mix_ratio).replace('.','_') )                     
        

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
                                Sol = [Sol, Sol]
                                L_Output.Export_Solution(Sol, Context, Band, 'S_'+str(km)+'_'+Band+'_'+Method+str(n_img)+'_'+str(n_obs)+'_'+str(Obs_std).replace('.','_') + "_"  +str(mix_ratio).replace('.','_') )

                        
                        

df.to_csv(csv_dir + csv_name + '.csv')

sys.exit(0)            

#dg=df.copy()
dg = pd.read_csv(csv_dir + csv_name + ".csv")
#dg.groupby(['Method','Band','Pol']).mean()['RMSE']
#dg = dg.groupby(['CellSize', 'Band', 'Method', 'mix_ratio']).mean()[['RMSE','RMSE_C0','RMSE_L0','RMSE_C1','RMSE_L1']]
dg = dg.groupby(['CellSize', 'Band', 'Method', 'mix_ratio'])['RMSE'].mean()
for MAPA in MAPAS:
    km=int(MAPA[4:6])
    print("Mapa: " + MAPA)
    for Band in Bands:
        print("Banda: " + Band)
        fig, ax = plt.subplots()
        #plt.ylim((0,10))
        for Method in Methods:
            serie = []
            for mix_ratio in mix_ratios:
                serie.append(dg[km, Band, Method, mix_ratio])
            ax.plot(mix_ratios, serie, label=Method)
        legend = ax.legend(loc='best', shadow=True, fontsize='x-large') 
        plt.title("Banda: " + Band + " Base Type: " + base_type)
        plt.savefig(csv_dir + "Imgs/" + Band + "_" + base_type + ".jpg")
        plt.show()
        

"""
km = 25
Band = "C"
fig, ax = plt.subplots()
plt.ylim((0,10))
for Method in Methods:
    serie = []
    for mix_ratio in mix_ratios:
        serie.append(dg[km, Band, Method, mix_ratio])
    ax.plot(mix_ratios, serie, label=Method)
legend = ax.legend(loc='best', shadow=True, fontsize='x-large') 
plt.show()
"""

                
        


#%%


sys.exit(0)

L_Output.Export_Solution(Sol, Context, Band, 'TestRafa')

Reg=np.array(Context['Dict_Vars']['SqCell'])
IReg=np.array([not r for r in Context['Dict_Vars']['SqCell']]) #irregular cell, frontier of LTs.
Mar=np.array(Context['Dict_Vars']['Margin']) #a marginal cell, border of the image
NMar=np.array([not m for m in Context['Dict_Vars']['Margin']]) #interior cell

SolN=Sol.copy()
SolN[Reg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_Costa')

SolN=SolH.copy()
SolN[IReg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_Reg')

SolN=SolH.copy()
SolN[Mar]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_NMar')

SolN=SolH.copy()
SolN[NMar]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_Mar')

SolN=SolH.copy()
SolN[Mar]=0
SolN[Reg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_CostaInterior')

SolN=SolH.copy()
SolN[Mar]=0
SolN[IReg]=0
L_Output.Export_Solution([SolN,SolN], Context, Band, 'TestRafa_LandInterior')

#%%


Reg=np.array(Context['Dict_Vars']['SqCell'])
IReg=np.array([not r for r in Context['Dict_Vars']['SqCell']]) #irregular cell, frontier of LTs.
Mar=np.array(Context['Dict_Vars']['Margin']) #a marginal cell, border of the image
NMar=np.array([not m for m in Context['Dict_Vars']['Margin']]) #interior cell
VType=Context['Dict_Vars']['VType']
L0=np.array(VType==0)
L1=np.array(VType==1)

B0=(NMar*IReg*L0)
B1=(NMar*IReg*L1)
I0=(NMar*Reg*L0)
I1=(NMar*Reg*L1)

print(B0.sum())
print(B1.sum())
print(I0.sum())
print(I1.sum())

#%%
def Metricas(Orig, Sol, Context, Margin=None, Regular=None, LT=None):
    n_Vars=Context['Dict_Vars']['n_Vars']    
    T=np.ones(n_Vars,dtype=bool)
    if (Margin==None):
        M=T
    elif Margin:
        M=np.array(Context['Dict_Vars']['Margin']) #a marginal cell, border of the image
    else:
        M=np.array([not m for m in Context['Dict_Vars']['Margin']]) #interior cell
    if (Regular==None):
        R=T
    elif Regular:
        R=np.array(Context['Dict_Vars']['SqCell']) #a square, regular cell, typically not land type frontier.
    else:
        R=np.array([not r for r in Context['Dict_Vars']['SqCell']]) #irregular cell, frontier of LTs.
    if (LT==None):
        L=T
    else:
        VType=Context['Dict_Vars']['VType']
        L=np.array(VType==LT)
    Cells=np.where(M*R*L)[0]
    
    return RMSE(Orig[Cells], Sol[Cells]),CC(Orig[Cells], Sol[Cells])




#%%
for sdev in [0.2,0.50,0.7,1,1.5]:
  for alpha in [.1,.25,.33,.45]: 
    print(sdev,alpha)
    Tb_base = L_Syn_Img.Create_Bimodal_Synth_Img(Context, mu=180, sdev=sdev, alpha=alpha)
    a = np.hstack(Tb_base)
    _ = plt.hist(a, bins=20) 
    plt.show() 
#%%

sdev=0.5
alpha=0.33
Tb_base = L_Syn_Img.Create_Bimodal_Synth_Img(Context, mu=180, sdev=sdev, alpha=alpha)
a = np.hstack(Tb_base)
_ = plt.hist(a, bins=20) 
plt.show() 
L_Output.Export_Solution([Tb_base,Tb_base], Context, "RealBase", 'RealBase' )                                 



import L_Syn_Img
Tb_base = L_Syn_Img.Create_Bimodal_Synth_Img(Context, mu=180)
a = np.hstack(Tb_base)
_ = plt.hist(a, bins=40) 
plt.show() 
L_Output.Export_Solution([Tb_base,Tb_base], Context, "RealBase", 'RealBase' )                                 




# BIMODAL
import L_Syn_Img
import L_Context
import L_Output
import numpy as np
import pandas as pd
import time
import pickle
import sys
import matplotlib.pyplot as plt
MAPA='Reg25' # Solo para bimodal.
BASE_DIR="/home/lbali/PMW-Tychonov/"
csv_dir=BASE_DIR + 'Output/'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
Tb_base = L_Syn_Img.Create_Bimodal_Synth_Img(Context, mu=180, sdev=0.3)
a = np.hstack(Tb_base)
_ = plt.hist(a, bins=40) 
plt.show() 
L_Output.Export_Solution([Tb_base,Tb_base], Context, "RealBase", 'RealBase' )                                 


# TRIGONOMETRIC
import L_Syn_Img
import L_Context
import L_Output
import numpy as np
import pandas as pd
import time
import pickle
import sys
import matplotlib.pyplot as plt
MAPA='LCA_25000m' # Solo para bimodal.
BASE_DIR="/home/lbali/PMW-Tychonov/"
csv_dir=BASE_DIR + 'Output/'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA
NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
Tb_base = L_Syn_Img.Create_Trig_Synth_Img(Context)
a = np.hstack(Tb_base)
_ = plt.hist(a, bins=40) 
plt.show() 
L_Output.Export_Solution([Tb_base,Tb_base], Context, "RealBase", 'RealBase' )                                 


# Random 180 para agua y random media 270 para tierra, es la situacion "ideal" para
# el EM, que supone guassianidad INDEOENDIENTE en cada tipo de land type.
# La estructura de correlaciòn espacial NO està incorporado en su modelo.
# Histogramas suponen en algùn sentido cierta iid, y no es lo que ocurre.
# No es la herramienta adecuada para visualizar resultados.

"""
El bimodal es "falso", el histograma no proviene de un sampleo iid sino que hay
correlaciòn espacial entre los elementos. No es una muestra en el sentido estricto de la
palabra, como para hablar de bimodalidad en la distribución.
Cada píxel tiene su propia distribución y hay correlación espacial.
El método EM debería primar en situaciones en donde hay IID normal en cada land type.
La correlación espacial rompe esas hipótesis, por otro lado la manera de medir el error
quizás debería ser relativo.

No se puede inferir la distribuciòn de densidad de los pìxeles tomando a ellos como muestra
pues no lo son, no provienen de un casi iid.

"""

