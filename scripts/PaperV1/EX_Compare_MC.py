# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:26:12 2017

@author: rgrimson
"""

from __future__ import print_function

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
import L_Syn_Img
import L_ParamEst
import numpy as np
import matplotlib.pyplot as plt


#%%
#PROCESS PARAMETERS
HDFfilename='GW1AM2_201207310439_227D_L1SGBTBR_2210210'
#Bands=['ka','K','ku','x','c']
Band='ka'
Obs_error_std=1
NeighborsBehaviour = L_Context.LOAD_NEIGHBORS
Methods = ["Rodgers_IT"]#"Weights", "Global_GCV_Tichonov"]#, "K_Global_GCV_Tichonov","Rodgers_IT",'LSTSQR','BGF']
N_rep=5#3
#%%

def main(): 
    #%%###################
    ## PREPROCESSING    
    
    #Load Context    
    Context=L_Context.Load_Context(wdir, NeighborsBehaviour)
    #Compute cell in the grid considered as MARGIN-CELLS (for correct error estimation)
    L_Syn_Img.Set_Margins(Context)
    Observation, tita=L_ReadObs.Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True)

    TITA_Synth={}    
    TITA={}
    RMSE={}
    
    for rep in range(N_rep):
        #Create Synthetic Image 
        TbH=L_Syn_Img.Create_Random_Synth_Img(Context)#Create_Trig_Synth_Img(Context)
        TbV=L_Syn_Img.Create_Trig_Synth_Img(Context)
        
        #Export Synthetic Image Map
        L_Output.Export_Solution([TbH,TbV], Context, "Synth", "SynthImg_r"+str(rep))
        
        #%
        s = ""
        #Load ellipses
        #Set Synthetic Obsetvations
        Observation=L_Syn_Img.Simulate_PMW(TbH,TbV,Context,Observation,Obs_error_std=Obs_error_std) #Simulate SynObs


        TITA_Synth[rep]={}
        TITA[rep]={}
        RMSE[rep]={}
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        TITA_Synth[rep]=copy.deepcopy(tita)
        
        tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
        s+= L_ParamEst.toString(tita, tita, 'Synthetic Image')
        
        
        for Method in Methods:        
            print("\n·································································")            
            print("........... SOLVING BAND %s with Method %s" %(Band, Method))
            s += "............. SOLVING BAND: " + str(Band) + "\n"
            s += "Method: " + str(Method) + "\n"
            s += ".............................................\n"

            #%##################################################
            ## SOLVE SYSTEM with synth sample parameters as input
            tita=L_ParamEst.recompute_param([TbH,TbV],Observation,tita)
            Sol=L_Solvers.Solve(Context,Observation,Method, tita)
        
            #%##################################################
            ## Export solution
            L_Output.Export_Solution(Sol, Context, Band, "SolSynthImg_%s_r%d"%(Method,rep))
            #%    
            #Compare Original and Reconstructed: Evaluate error (with and without margins)
            print("\nSummary:\n--------")
            titaReal=copy.deepcopy(L_ParamEst.recompute_param([TbH,TbV],Observation,tita))
            L_ParamEst.pr(titaReal,'Synthetic Image')
            
            tita=L_ParamEst.recompute_param(Sol,Observation,tita)
            s += L_ParamEst.toString(tita, titaReal, 'Reconstructed Image_'+str(rep))
            L_ParamEst.pr(tita, 'Reconstructed Image')
        
            print("\nErrors:\n--------")
            print("RMSE H: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbH,Sol[0],Context), L_Syn_Img.RMSE(TbH,Sol[0])))
            print("RMSE V: %.3f,%.3f"%(L_Syn_Img.RMSE_M(TbV,Sol[1],Context), L_Syn_Img.RMSE(TbV,Sol[1])))
            s += "RMSE H: " + str(L_Syn_Img.RMSE_M(TbH,Sol[0],Context)) + "\n"
            s += "RMSE V: " + str(L_Syn_Img.RMSE_M(TbV,Sol[1],Context)) + "\n"
            RMSE[rep][Method]={}
            RMSE[rep][Method]['H']=L_Syn_Img.RMSE_M(TbH,Sol[0],Context)
            RMSE[rep][Method]['V']=L_Syn_Img.RMSE_M(TbV,Sol[1],Context)
            TITA[rep][Method]=copy.deepcopy(tita)
    print("RESULTADOS FINALES \n\n\n")
    print(s)
    print("Generando archivo...")
    f = open(wdir + '/Salidas/resultadosCorrida.txt', 'w')
    f.write(s) # Ahora si anda, no le gustaba el nombre solo del archivo, por algun motivo...
    f.close()
#%%            

        
#%%
def plot_scat(RMSE, TITA, TITA_Synth, Bands=None, Methods=None, Pols=None):
    #%%
    #initialize    
    if (Bands==None):
      Bands=list(TITA.keys())
    nBands=len(Bands)
    

    if (Methods==None):
        Methods=list(TITA[Bands[0]].keys())
    
    if (Pols==None):
        Pols=['H','V']
    nPols=len(Pols)
    
    RMSExS2      = np.zeros([2])
    RMSExS2_Mthd = {}
    
    #reorder
    for Method in Methods:
        RMSExS2_Mthd[Method]={}
        for pol in Pols:
            RMSExS2_Mthd[Method][pol]={}
            for Band in Bands:
                rmse=RMSE[Band][Method][pol]
                nstdM=np.max([np.max(np.array(list(map(np.sqrt,np.abs(TITA[Band][Method]['H']['sigma2']))))/np.array(list(map(np.sqrt,TITA_Synth[Band]['H']['sigma2'])))),                 np.max(np.array(list(map(np.sqrt,np.abs(TITA[Band][Method]['V']['sigma2']))))/np.array(list(map(np.sqrt,TITA_Synth[Band]['V']['sigma2']))))])
                nstdm=np.min([np.min(np.array(list(map(np.sqrt,np.abs(TITA[Band][Method]['H']['sigma2']))))/np.array(list(map(np.sqrt,TITA_Synth[Band]['H']['sigma2'])))),                 np.min(np.array(list(map(np.sqrt,np.abs(TITA[Band][Method]['V']['sigma2']))))/np.array(list(map(np.sqrt,TITA_Synth[Band]['V']['sigma2']))))])
                if (1/nstdm)>nstdM:
                    nstd=nstdm
                else:
                    nstd=nstdM
                
                RMSExS2[0]=rmse
                RMSExS2[1]=nstd #max quotient of std_lt_sym/std_lt_synth
                RMSExS2_Mthd[Method][pol][Band]=copy.deepcopy(RMSExS2)
    #%
    nM=len(Methods)
    RMSEv=np.zeros([nBands*nM*nPols])
    NSTDv=np.zeros([nBands*nM*nPols]) 
    COLS=np.zeros([nBands*nM*nPols],dtype=str)  
    MRKS=np.zeros([nBands*nM*nPols],dtype=str) 
    FS=np.zeros([nBands*nM*nPols],dtype=str) 
    Label=np.zeros([nBands*nM*nPols],dtype=str) 
    
    plt.clf()
    Colors=['y','b','r','m','c','k','g']
    import matplotlib.lines as mlines
    Markers=mlines.Line2D.filled_markers
    FaceStyles=['full','none','left','right','bottom','top']
    for iM in range(nM):
        for iBand in range(nBands):
            for iPol in range(nPols):            
                Method=Methods[iM]
                Band=Bands[iBand]
                pol=Pols[iPol]
                rmse=RMSExS2_Mthd[Method][pol][Band][0]
                nstd=RMSExS2_Mthd[Method][pol][Band][1]
                RMSEv[iM*nBands*nPols+iPol*nBands+iBand]=rmse
                NSTDv[iM*nBands*nPols+iPol*nBands+iBand]=nstd

                col=Colors[iM]
                mrk=Markers[iPol]
                fs=FaceStyles[iPol]
                s=20+iBand*70
                label=Method+'_'+pol+'_'+Band
                COLS[iM*nBands*nPols+iPol*nBands+iBand]=col
                MRKS[iM*nBands*nPols+iPol*nBands+iBand]=mrk
                FS[iM*nBands*nPols+iPol*nBands+iBand]=fs
                Label[iM*nBands*nPols+iPol*nBands+iBand]=label
                plt.scatter(rmse,np.log(nstd),c=col, marker=mrk, label=label,alpha=0.7,s=s)
#%%
#plot(RMSE,TITA,TITA_Synth)
    
#%%

def barplot():    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import plotly.plotly as py
    import plotly.tools as tls
    # Learn about API authentication here: https://plot.ly/python/getting-started
    # Find your api_key here: https://plot.ly/settings/api
    
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    
    N = 5
    menMeans = (20, 35, 30, 35, 27)
    womenMeans = (25, 32, 34, 20, 25)
    menStd = (2, 3, 4, 1, 2)
    womenStd = (3, 5, 2, 3, 3)
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    p1 = ax.bar(ind, menMeans, width, color=(0.2588,0.4433,1.0))
    p2 = ax.bar(ind, womenMeans, width, color=(1.0,0.5,0.62),
                 bottom=menMeans)
    ax.set_ylabel('Scores')
    ax.set_xlabel('Groups')
    ax.set_title('Scores by group and gender')
    
    ax.set_xticks(ind + width/2.)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    
    
    
    plotly_fig = tls.mpl_to_plotly( mpl_fig )
    
    # For Legend
    plotly_fig["layout"]["showlegend"] = True
    plotly_fig["data"][0]["name"] = "Men"
    plotly_fig["data"][1]["name"] = "Women"
    
    
    plot_url = py.plot(plotly_fig, filename='stacked-bar-chart')





if __name__ == "__main__":
    main()

