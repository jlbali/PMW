# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:21:01 2018

@author: rgrimson
"""



import my_base_dir
#Import libraries
import copy
import L_ReadObs
import L_Context
import L_Files
import L_Output
import L_Solvers
import L_ParamEst
import L_Syn_Img
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


#### Solvers ############

class Solver:
    
    def __init__(self):
        pass
    
    def solve(self, Context, Observation, extraParams = None):
        pass
    
    def get_name(self):
        return "UNDEFINED_SOLVER"
    
    def get_Sol(self):
        return self.Sol
    
    def get_RMSE(self, TbH, TbV, Context):
        rmseH=L_Syn_Img.RMSE_M(TbH,self.Sol[0],Context)
        rmseV=L_Syn_Img.RMSE_M(TbV,self.Sol[1],Context)
        return (rmseH, rmseV)
    
    def get_corr(self, TbH, TbV):
        corrH = Coef_correlacion(TbH,self.Sol[0])
        corrV = Coef_correlacion(TbV,self.Sol[1]) 
        return (corrH, corrV)



class LSTSQR(Solver):
    
    def __init__(self):
        pass
    
    def solve(self, Context, Observation, extraParams = None):
        neighbors = Context["Neighbors"]
        Wt=Observation['Wt']
        H = L_Solvers.constructH(Wt, neighbors)
        lamb = 0.0
        self.Sol = L_Solvers.Solve_Tichonov(Observation, lamb, H)        
        return self.Sol

    def get_name(self):
        return "LSTSQR"

class Weights(Solver):
    
    def __init__(self):
        pass
    
    def solve(self, Context, Observation, extraParams = None):
        self.Sol=L_Solvers.Solve_Weights(Context,Observation)
        return self.Sol
    
    def get_name(self):
        return "WEIGHTS"

class Tychonov(Solver):
    
    def __init__(self):
        pass
    
    def solve(self, Context, Observation, extraParams = None):
        gridLambs = L_Solvers.grillaLogaritmica(1e-6, 100.0, 9)
        Vtypes = Observation['VType']
        self.Sol = L_Solvers.Solve_Global_GCV_Tichonov(Vtypes, Observation, gridLambs)
        return self.Sol

    def get_name(self):
        return "TYCHONOV"
    
class STAT(Solver):
    
    def __init__(self,  verbose=True):
        #self.damp = damp
        self.verbose = verbose
    
    def solve(self, Context, Observation, extraParams = None):
        tita = Observation["tita"]
        damp = Observation["noise"]
        self.Sol=L_Solvers.Solve_Rodgers_IT(Context,Observation,tita,damp=damp,verbose=self.verbose)
        return self.Sol
        
    def get_name(self):
        return "STAT"

class BG(Solver):
    
    def __init__(self, Context, distances_path, gamma, omega, km = 25):
        self.gamma = gamma
        self.omega = omega
        #self.error_std = error_std
        self.dic_maxObs={'KA10':6,'KU10':6,'K10':6,'X10':4,'C10':4,'KA25':20,'KU25':20,'K25':20,'X25':12,'C25':12}
        self.km = km
        if not L_Files.exists(distances_path):
            distances = L_Solvers.precompute_all_optimized(Context)
            L_Solvers.save_distances(distances_path, distances)
        self.distances = L_Solvers.load_distances(distances_path)
    
    def solve(self, Context, Observation, extraParams = None):
        key = Observation["Band"] + str(self.km)
        max_obs = self.dic_maxObs[key]
        if extraParams == None:
            gamma = self.gamma
            omega = self.omega
            #error_std = self.error_std
        else:
            gamma = extraParams["gamma"]
            omega = extraParams["omega"]
            #error_std = extraParams["error_std"]
        error_std = Observation["noise"]
        BG_Coef = L_Solvers.BG_Precomp_MaxObs_Cell(self.distances, Context, Observation, gamma=gamma,w=omega,error_std=error_std,MaxObs=max_obs)
        self.Sol = [BG_Coef.dot(Observation['Tb']['H']), BG_Coef.dot(Observation['Tb']['V'])]
        return self.Sol

    def get_name(self):
        return "BG"



######################### EVALUACION ####################################

def Coef_correlacion(img_sintetica,img_solucion): # Toma dos imagenes, cada imagen es un array
    matriz_correlacion = np.corrcoef(img_sintetica,img_solucion)
    return matriz_correlacion[0,1]

##########################################################################



class Reporte:
    
    def __init__(self):
        self.lineas = []
    
    def add_linea(self, rep, band, noise, method, H_RMSE, V_RMSE, H_corr, V_corr):
        linea = {
                "rep": rep,
                "band": band,
                "noise": noise,
                "method": method,
                "H_RMSE": H_RMSE,
                "V_RMSE": V_RMSE,
                "H_corr": H_corr,
                "V_corr:": V_corr
        }
        self.lineas.append(linea)

    def get_linea(self, rep, band, noise, method):
        for linea in self.lineas:
            if linea["rep"] == rep and linea["band"] == band and linea["noise"] == noise and linea["method"] == method:
                    return linea
        return None

    def get_means(self, band, noise, method):
        h_rmse = []
        v_rmse = []
        h_corr = []
        v_corr = []
        for linea in self.lineas:
            if linea["band"] == band and linea["noise"] == noise and linea["method"] == method:
                    h_rmse.append(linea["H_RMSE"])
                    v_rmse.append(linea["V_RMSE"])
                    h_corr.append(linea["H_corr"])
                    v_corr.append(linea["B_corr"])
        return {
                "H_RMSE": np.mean(h_rmse),
                "V_RMSE": np.mean(v_rmse),
                "H_corr": np.mean(h_corr),
                "V_corr": np.mean(v_corr)
        }
            
    def get_sd(self, band, noise, method):
        h_rmse = []
        v_rmse = []
        h_corr = []
        v_corr = []
        for linea in self.lineas:
            if linea["band"] == band and linea["noise"] == noise and linea["method"] == method:
                    h_rmse.append(linea["H_RMSE"])
                    v_rmse.append(linea["V_RMSE"])
                    h_corr.append(linea["H_corr"])
                    v_corr.append(linea["V_corr"])
        return {
                "H_RMSE": np.std(h_rmse),
                "V_RMSE": np.std(v_rmse),
                "H_corr": np.std(h_corr),
                "V_corr": np.std(v_corr)
        }
    
    def save_file(self,path):
        f = open(path, "w")
        s = "Rep\tBand\tNoise\tMethod\tH_RMSE\tV_RMSE\tH_corr\tV_corr\n"
        for linea in self.lineas:
            s += str(linea["rep"]) + "\t" + str(linea["band"]) + "\t" + str(linea["noise"]) + "\t"
            s += str(linea["method"]) + "\t" + str(linea["H_RMSE"]) + "\t" + str(linea["V_RMSE"]) + "\t"
            s += str(linea["H_corr"]) + "\t" + str(linea["V_corr"]) + "\n"
        f.write(s)
        f.close()
    
    def load_file(self,path):
        self.lineas = []
        f = open(path)
        lines = f.readlines()
        for i in range(1, len(lines)):
            tokens = lines[i].split("\t")
            self.add_linea(int(tokens[0]), tokens[1], float(tokens[2]), tokens[3], float(tokens[4]), float(tokens[5]), float(tokens[6]), float(tokens[7]))
        f.close()
        
###################################################

def start_simulation():
    # Caracteristicas del Mapa    
    BASE_DIR = my_base_dir.get()
    print("Base DIR: ", BASE_DIR)
    MAPA='Ajo_25km'
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context) # Puede que esto de los margenes genere problemas...
    # Caracteristicas de la antena.
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
    Bands=['KA','K','KU','C','X']
    # Inicializacion de los metodos.
    lstsqr = LSTSQR()
    weights = Weights()
    tychonov = Tychonov()
    stat = STAT()
    bg = BG(Context, BASE_DIR + "data25km.pickle", gamma=np.pi/4.0, omega = 0.001, km = 25)    
    methods = [lstsqr, weights, tychonov, stat, bg]
    # Caracteristicas de la simulacion.
    replicaciones = 1
    noises = np.linspace(1.0, 3.0, 3)
    # EMPIEZAN LAS SIMULACIONES
    np.random.seed(40)
    reporte = Reporte()
    for rep in range(replicaciones):
        print("Replicacion :", rep)        
        # Generacion imagen sintetica.
        TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
        TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
        for band in Bands:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            print("Banda: ", band)            
            # Generamos la observacion satelital.            
            Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,band,GC=True)
            for noise in noises:
                print("Ruido: ", noise)                
                # Contaminamos la observacion.
                tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita) # El tita se basa en los valores verdaderos??? Parece medio trampa...           
                Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=noise) 
                Observation["tita"] = tita
                Observation["noise"] = noise
                for method in methods:
                    print("Metodo: ", method.get_name())                    
                    method.solve(Context, Observation)
                    rmse = method.get_RMSE(TbH, TbV, Context)
                    corr = method.get_corr(TbH, TbV)
                    print("RMSE: ", rmse)
                    print("Corr: ", corr)                                                                                               
                    reporte.add_linea(rep, band, noise, method.get_name(), rmse[0], rmse[1], corr[0], corr[1])
    return reporte
                                                                                                                                                                                    
# Oko que para BG y STAT es preciso pasarle el noise level del momento.
# O convendra hacer el estudio de robustez con un noise level incorrecto?
    
        
# Ver de mejorar el tema del tita. 

#reporte = start_simulation()





#################################### BUSQUEDA DEL GAMMA PARA BACKUS-GILBERT ################################
    
def search_BG_gamma():
    band = "KU"
    #gammas = np.linspace(0, np.pi/2, 10)
    #gammas = np.linspace(0, np.pi/2, 20)
    gammas = np.linspace(0, 0.4, 10)
    f = open("/home/lbali/reporteGammaKu_2.txt", "w")
    omega = 0.001
    BASE_DIR = my_base_dir.get()
    print("Base DIR: ", BASE_DIR)
    MAPA='Ajo_25km'
    wdir=BASE_DIR + 'Mapas/%s/'%MAPA
    NeighborsBehaviour = L_Context.COMPUTE_IF_NECESSARY
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    L_Syn_Img.Set_Margins(Context) # Puede que esto de los margenes genere problemas...
    # Caracteristicas de la antena.
    HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
    noise = 1.0  
    np.random.seed(40)
    TbH = L_Syn_Img.Create_Random_Synth_Img(Context)
    TbV = L_Syn_Img.Create_Trig_Synth_Img(Context)
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,band,GC=True)
    tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita) # El tita se basa en los valores verdaderos??? Parece medio trampa...           
    Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=noise) 
    Observation["tita"] = tita
    Observation["noise"] = noise
    f.write("gamma\tRMSE_H\tRMSE_V\n")
    rmse_H = []
    rmse_V = []
    for gamma in gammas:
        bg = BG(Context, BASE_DIR + "data25km.pickle", gamma=gamma, omega = omega, km = 25  )        
        bg.solve(Context, Observation)
        rmse = bg.get_RMSE(TbH, TbV, Context)
        corr = bg.get_corr(TbH, TbV)
        print("gamma: ", gamma)
        print("RMSE: ", rmse)
        f.write(str(gamma) + "\t" + str(rmse[0]) + "\t" + str(rmse[1]) + "\n")
        rmse_H.append(rmse[0])
        rmse_V.append(rmse[1])
    plt.plot(gammas, rmse_H)
    plt.show()
    plt.plot(gammas, rmse_V)
    plt.show()
    f.close()




def test():
    reporte = start_simulation()
    reporte.save_file("/home/lbali/reporte.txt")

#test()

def test2():
    search_BG_gamma()


test2()

"""
- Tema con el tita que parece el verdadero y lo usa el metodo STAT.
- Tema de shape en el backus-gilbert.

omega = 0.001 (sugerido por Long 1998)

Best gamma:

Banda KA, Random (H) --> 0.050
Banda KA, Trig (V) -->   0.050

Banda K, Random (H) --> 0.050
Banda K, Trig (V) --> 0.10 (podria ser 0.05 tambien, no cambia mucho)

Banda KU, Random (H) --> 0.0 (es monotona)
Banda KU, Trig (V) --> 0.1 (podria ser 0.05 tambien, no cambia mucho)

Quedan bandas C y X para realizar.

"""

