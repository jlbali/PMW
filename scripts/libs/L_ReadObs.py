# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 13:27:55 2017

@author: rgrimson
"""
from __future__ import print_function
import numpy as np
from numpy import linalg as la
import utm
import h5py as hdf
#uncomment to read AMSRE files
#from pyhdf.SD import SD, SDC
import sys
import L_GeoCorr
import L_Files
import L_Kernels as L_Kernels
import L_Context
import logging
import time


SAVE_ELLS=True #file with observations for the study region, in internal format
SAVE_OBS=True  #file with precomputed Kernels for each band

#import my_base_dir




#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Load and Save Observations                                               %#
#%                                                                           %#
##%############################################################################
#%%
def Load_SingleBand_Ells(HDFfilename, Context,Band):
    path=Context['wdir']
    if HDFfilename.find('L1SGBTBR')>=0:
        Dict_Bands=Read_HDF_AMSR2_L1B(path+'HDF/',HDFfilename,[Band],Context['WA_Cords'])
    elif HDFfilename.find('L1SGRTBR')>=0:
        Dict_Bands=Read_HDF_AMSR2_L1R(path+'HDF/',HDFfilename,[Band],Context['WA_Cords'])
    elif HDFfilename.find('AMSR_E')>=0:    
        Dict_Bands=Read_HDF_AMSR_E_L2A(path+'HDF/',HDFfilename,[Band],Context['WA_Cords'])
    else:
        print("Cannot guess Satellite or Processing Level.")
        print("Assuming AMSR-2 L1B file")
        Dict_Bands=Read_HDF_AMSR2_L1B(path+'HDF/',HDFfilename,[Band],Context['WA_Cords'])
        
    L_GeoCorr.GeoCorrect_Band(Dict_Bands[Band],Context['WA_Cords'],Context['Dict_Grids'])
    return Dict_Bands


def Load_HDF_Ells(HDFfilename, Context,GC=True):
    path=Context['wdir']
    if HDFfilename.find('L1SGBTBR')>=0:
        Dict_Bands=Read_HDF_AMSR2_L1B(path+'HDF/',HDFfilename,Context['Bands'],Context['WA_Cords'])
    elif HDFfilename.find('L1SGRTBR')>=0:
        Dict_Bands=Read_HDF_AMSR2_L1R(path+'HDF/',HDFfilename,Context['Bands'],Context['WA_Cords'])
    elif HDFfilename.find('AMSR_E')>=0:    
        Dict_Bands=Read_HDF_AMSR_E_L2A(path+'HDF/',HDFfilename,Context['Bands'],Context['WA_Cords'])
    else:
        print("Cannot guess Satellite or Processing Level.")
        print("Assuming AMSR-2 L1B file")
        Dict_Bands=Read_HDF_AMSR2_L1B(path+'HDF/',HDFfilename,Context['Bands'],Context['WA_Cords'])
        
    
    if GC:
        L_GeoCorr.GeoCorrectBands(Dict_Bands,Context['Dict_Grids'])
    else:
        #print(Dict_Bands)
        for Band in Context['Bands']:
            Dict_Bands[Band]['GeoCorrection']={'X':0.0,'Y':0.0,'OK':True}
            

    return Dict_Bands
#%%
def Save_PKL_Ells(Dict_Bands,HDFfilename, Context):
    path=Context['wdir']
    L_Files.save_obj(Dict_Bands, path+'GC/'+HDFfilename)

    
#%%    
def Load_PKL_Ells(HDFfilename,Context):
    path=Context['wdir']
    return L_Files.load_obj(path+'GC/'+HDFfilename)
#%%
def Load_PKL_or_HDF_Ells(HDFfilename, Context,GC=True):
    path=Context['wdir']
    pklz_fn=path+'GC/'+HDFfilename+'.pklz'
    if L_Files.exists(pklz_fn):
        print('##   Using preprocessed HDF file')
        Dict_Bands=Load_PKL_Ells(HDFfilename,Context)
    else:
        print('##   Processing HDF file')
        print("SaveElls", SAVE_ELLS)
        Dict_Bands=Load_HDF_Ells(HDFfilename, Context,GC=GC)
        print("Generado los Dict_Bands")
        if SAVE_ELLS:
            Save_PKL_Ells(Dict_Bands,HDFfilename, Context)
    return Dict_Bands
    
    
#%%
def Save_PKL_Obs(Dict_Bands,HDFfilename, Context,Band):
    path=Context['wdir']
    Band=Band.upper()
    L_Files.save_obj(Dict_Bands, path+'GC/'+HDFfilename+'_'+Band+'Ker')

    
#%%    
def Load_PKL_Obs(HDFfilename,Context,Band):
    path=Context['wdir']
    Band=Band.upper()
    return L_Files.load_obj(path+'GC/'+HDFfilename+'_'+Band+'Ker')
    
#%%
#MAIN LOAD FUNCTION
def Load_Shifted_Band_Kernels(HDFfilename,Context,Band,Shift_X, Shift_Y):
    path=Context['wdir']
    Band=Band.upper()
    pklz_fn=path+'GC/'+HDFfilename+'_'+Band+'Ker'+'.pklz'
    MSG="*******************************************************************************\n"
    Dict_Bands=Load_SingleBand_Ells(HDFfilename, Context,Band)
    Dict_Bands[Band]['GeoCorrection']['X']+=Shift_X
    Dict_Bands[Band]['GeoCorrection']['Y']+=Shift_Y
    
    print('######################################')
    print('##   PRECOMPUTING K for BAND',Band,'#####')
    print('######################################')
    MSG+='Computing Kernels for Band' + Band
    Observation, tita = L_Kernels.Compute_Kernels(Dict_Bands, Band, Context)
            
    #LOG
    localtime = time.asctime( time.localtime(time.time()) ) + '\n'
    MSG+='\n'
    if Observation=={}:
        MSG+="Error loading " + HDFfilename + '\n'        
    else:
        MSG+="Loaded " + HDFfilename + '\n' 
        MSG+= (not Observation['GeoCorrection']['OK'])*'FAILED GeoCorrecting: ' + Observation['GeoCorrection']['OK']*'GeoCorrection: '
        MSG+='[' + str(Observation['GeoCorrection']['X']) +', ' + str(Observation['GeoCorrection']['Y']) + ']\n'
        MSG+= 'LSQRSols: H ' + str(Observation['LSQRSols']['H']['Sol']) + ' Error:' + str(Observation['LSQRSols']['H']['norm']) + '\n'
        MSG+= '          V ' + str(Observation['LSQRSols']['V']['Sol']) + ' Error:' + str(Observation['LSQRSols']['V']['norm']) + '\n'
    logging.info(localtime + MSG)
    return Observation, tita 
    
def Load_Band_Kernels(HDFfilename,Context,Band):
    path=Context['wdir']
    Band=Band.upper()
    pklz_fn=path+'GC/'+HDFfilename+'_'+Band+'Ker'+'.pklz'
    MSG="*******************************************************************************\n"
    if L_Files.exists(pklz_fn):
        print('##   Using precomputed kernels for band',Band)
        Observation, tita=Load_PKL_Obs(HDFfilename, Context, Band)
        MSG+='From PKL'
    else:
        Dict_Bands=Load_SingleBand_Ells(HDFfilename, Context,Band)
        print('######################################')
        print('##   PRECOMPUTING K for BAND',Band,'#####')
        print('######################################')
        MSG+='Computing Kernels for Band ' + Band
        Observation, tita = L_Kernels.Compute_Kernels(Dict_Bands, Band, Context)
            
            
        if SAVE_OBS:
            MSG+='\nSaving PKL'
            Save_PKL_Obs([Observation, tita], HDFfilename, Context, Band)
            
    #LOG
    localtime = time.asctime( time.localtime(time.time()) ) + '\n'
    MSG+='\n'
    if Observation=={}:
        MSG+="Error loading (failed to geocorrect)" + HDFfilename + '\n'        
    else:
        MSG+="Loaded " + HDFfilename + '\n' + "Band: " + Band + '\n' 
        MSG+= (not Observation['GeoCorrection']['OK'])*'FAILED GeoCorrecting: ' + Observation['GeoCorrection']['OK']*'GeoCorrection: '
        MSG+='[' + str(Observation['GeoCorrection']['X']) +', ' + str(Observation['GeoCorrection']['Y']) + ']\n'
        MSG+= 'LSQRSols: H ' + str(Observation['LSQRSols']['H']['Sol']) + ' Error:' + str(Observation['LSQRSols']['H']['norm']) + '\n'
        MSG+= '          V ' + str(Observation['LSQRSols']['V']['Sol']) + ' Error:' + str(Observation['LSQRSols']['V']['norm']) + '\n'
    logging.info(localtime + MSG)
    return Observation, tita 
#%%
def Load_PKL_or_Compute_Kernels(HDFfilename,Context,Band,GC=True):
    path=Context['wdir']
    Band=Band.upper()
    pklz_fn=path+'GC/'+HDFfilename+'_'+Band+'Ker'+'.pklz'
    MSG="*******************************************************************************\n"
    print("Presto a cargar para banda", Band)
    if L_Files.exists(pklz_fn):
        print('##   Using precomputed kernels for band',Band)
        Observation, tita=Load_PKL_Obs(HDFfilename, Context, Band)
        MSG+='From PKL'
    else:
        print("Preparando a cargar las cosas....")
        Dict_Bands=Load_PKL_or_HDF_Ells(HDFfilename, Context,GC=GC)
        print('######################################')
        print('##   PRECOMPUTING K for BAND',Band,'#####')
        print('######################################')
        MSG+='Computing Kernels for Band' + Band
        Observation, tita = L_Kernels.Compute_Kernels(Dict_Bands, Band, Context)
            
        if SAVE_OBS:
            MSG+='\nSaving PKL'
            Save_PKL_Obs([Observation, tita], HDFfilename, Context, Band)
    localtime = time.asctime( time.localtime(time.time()) ) + '\n'
    MSG+='\n'
    if Observation=={}:
        MSG+="Error loading " + HDFfilename + '\n'        
    else:
        MSG+="Loaded " + HDFfilename + '\n'  + "Band: " + Band + '\n' 
        MSG+= (not Observation['GeoCorrection']['OK'])*'FAILED GeoCorrecting: ' + Observation['GeoCorrection']['OK']*'GeoCorrection: '
        MSG+='[' + str(Observation['GeoCorrection']['X']) +', ' + str(Observation['GeoCorrection']['Y']) + ']\n'
        MSG+= 'LSQRSols: H ' + str(Observation['LSQRSols']['H']['Sol']) + ' Error:' + str(Observation['LSQRSols']['H']['norm']) + '\n'
        MSG+= '          V ' + str(Observation['LSQRSols']['V']['Sol']) + ' Error:' + str(Observation['LSQRSols']['V']['norm']) + '\n'
    logging.info(localtime + MSG)
    return Observation, tita 
#%%
def Load_PKL_or_Compute_Kernels_With_GC_From_Other_Map(HDFfilename,Context,Band,MapPath):
    Band=Band.upper()
    path=Context['wdir']
    pklz_fn=path+'GC/'+HDFfilename+'_'+Band+'Ker'+'.pklz'
    pklz2_fn=MapPath+'GC/'+HDFfilename+'_'+Band+'Ker'+'.pklz'
    if L_Files.exists(pklz_fn):
        print('##   Using precomputed kernels for band',Band)
        Observation, tita=Load_PKL_Obs(HDFfilename, Context, Band)
    else:
        #read geocorrection from other map
        print('##   Reading GeoCorr from %s for Band %s'%(MapPath,Band))
        Context2=L_Context.Load_Context(MapPath, L_Context.NO_NEIGHBORS)
        Dict_Bands2=Load_PKL_or_HDF_Ells(HDFfilename, Context2)

        #read data and copy geocorrection
        print('##   Reading Ells')
        Dict_Bands=Load_PKL_or_HDF_Ells(HDFfilename, Context,GC=False)
        Dict_Bands[Band]['GeoCorrection'] = Dict_Bands2[Band]['GeoCorrection'] 
        
        #compute and save kernels
        Observation, tita = L_Kernels.Compute_Kernels(Dict_Bands, Band, Context,Mode='Cell')
        if SAVE_OBS:
            Save_PKL_Obs([Observation, tita], HDFfilename, Context, Band)
    return Observation, tita 

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   AMSR-2 L1B product reader                                               %#
#%                                                                           %#
##%############################################################################
def Read_HDF_AMSR2_L1B(path,filename,Bands,WA_Cords):
    Bands=[x.upper() for x in Bands]
    mLat=WA_Cords['mLat']
    mLon=WA_Cords['mLon']
    MLat=WA_Cords['MLat']
    MLon=WA_Cords['MLon']
    if ((filename.find('D_')*filename.find('A_'))>=0):
        exit()
    elif filename.find('D_')>0:
        PASS='D'
    else:
        PASS='A'
    inf=path+filename+'.h5'
    print (inf)
    infile = hdf.File(inf, "r")

    # Get the data
    Lat89A = infile["/Latitude of Observation Point for 89A"][:][:]
    Lon89A = infile["/Longitude of Observation Point for 89A"][:][:]
    
    print ("-----------------------------------------------------")
    print ("Reading HDF:"+filename)
    print ("Pass:", PASS)

    #Half power diameter for each band
    HPDMs={'C':62000,'X':42000,'KU':22000,'K':26000,'KA':12000}   #major diam in m
    HPDms={'C':35000,'X':24000,'KU':14000,'K':15000,'KA':7000}    #minor diam in m
    #A1 #6G-1.16934, #7G-0.86160, #10G-1.04596, #18G-1.08919, #23G-1.08342, #36G-0.80741
    #A2 #6G--0.03576, #7G--0.04742, #10G--0.20515, #18G-0.01587, #23G--0.06023, #36G-0.05469
    A1s={'C':0.86160,'X':1.04596,'KU':1.08919,'K':1.08342,'KA':0.80741}
    A2s={'C':-0.04742,'X':-0.20515,'KU':0.01587,'K':-0.06023,'KA':0.05469}
    
    D=Lat89A.min(axis=1)
    W=np.where((D>mLat)&(D<MLat))[0]
    y1minLat=W.min()
    y1maxLat=W.max()
    
    D=Lat89A.max(axis=1)
    W=np.where((D>mLat)&(D<MLat))[0]
    y2minLat=W.min()
    y2maxLat=W.max()
    
    ymin=min(y1minLat,y2minLat)
    ymax=max(y1maxLat,y2maxLat)+1
    
    D=Lon89A[ymin:ymax].min(axis=0)
    W=np.where((D>mLon)&(D<MLon))[0]
    x1minLon=W.min()
    x1maxLon=W.max()
    
    D=Lon89A[ymin:ymax].max(axis=0)
    W=np.where((D>mLon)&(D<MLon))[0]
    x2minLon=int(W.min()/2)*2     #so its always even
    x2maxLon=int(W.max()/2)*2+1   #so its always odd
    
    xmin=min(x1minLon,x2minLon)
    xmax=max(x1maxLon,x2maxLon)+1
    
    Lat89A =  Lat89A[ymin:ymax,xmin:xmax]
    Lon89A =  Lon89A[ymin:ymax,xmin:xmax]
    Tbh = dict()
    Tbv = dict()
    #Get data
    Tbh['KA']  =  infile["/Brightness Temperature (36.5GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['KA']  =  infile["/Brightness Temperature (36.5GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbh['K']   =  infile["/Brightness Temperature (23.8GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['K']   =  infile["/Brightness Temperature (23.8GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbh['KU']  =  infile["/Brightness Temperature (18.7GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['KU']  =  infile["/Brightness Temperature (18.7GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbh['X']   =  infile["/Brightness Temperature (10.7GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['X']   =  infile["/Brightness Temperature (10.7GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbh['C']   =  infile["/Brightness Temperature (7.3GHz,H)"] [(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['C']   =  infile["/Brightness Temperature (7.3GHz,V)"] [(ymin):(ymax),int(xmin/2):int(xmax/2)]

    Dict_Bands={}
    Dict_Bands['pass']=PASS
    for i in range(len(Bands)):
        Band=Bands[i]
        #outWM=path+'Dict/'+infilenm+'_'+Band #working area
        print ("  Reading band:"+Band+',',  end=" ")
        sys.stdout.flush()
        
        n_ell=0
        Dict_Band={}
        Dict_Ells={}
        Dict_Band['Band']=Band
        Dict_Band['HPDm']=HPDms[Band]
        Dict_Band['HPDM']=HPDMs[Band]
        #E=np.zeros([0,5])
        for y in range(Lat89A.shape[0]):
          for x in range(int(Lat89A.shape[1]/2)):
            #Write ellipses info to DICT
            m1 = np.array([ Lat89A[y,2*x],Lon89A[y,2*x]]) 
            m2 = np.array([ Lat89A[y,2*x+1],Lon89A[y,2*x+1]])
            #print m1,m2,
            P1=(np.cos(m1[1]*np.pi/180)*np.cos(m1[0]*np.pi/180),np.sin(m1[1]*np.pi/180)*np.cos(m1[0]*np.pi/180),np.sin(m1[0]*np.pi/180))
            P2=(np.cos(m2[1]*np.pi/180)*np.cos(m2[0]*np.pi/180),np.sin(m2[1]*np.pi/180)*np.cos(m2[0]*np.pi/180),np.sin(m2[0]*np.pi/180))
            ex=np.array(P1)
            C=np.cross(ex,np.array(P2))
            ez=C/la.norm(C)
            ey=np.cross(ez,ex)
            th=np.arccos(np.dot(P1,P2))
            
            
            A1=A1s[Band]
            A2=A2s[Band]
            nP= np.cos(A2*th)*np.cos(A1*th)*ex+np.cos(A2*th)*np.sin(A1*th)*ey+np.sin(A2*th)*ez
            nlat=np.arcsin(nP[2])
            nlon=180/np.pi*np.arcsin(nP[1]/np.cos(nlat))
            nlat=180/np.pi*nlat
            
            LocalNorth=np.dot(np.array([0,0,1]),ey)*ey+np.dot(np.array([0,0,1]),ez)*ez
            LocalNorth/=la.norm(LocalNorth)
            LocalWest=np.cross(ex,LocalNorth)
            nang=180/np.pi*np.arccos(np.dot(LocalNorth,ez))*np.sign(np.dot(LocalWest,ez))
            #if (nlat<MLat)&(nlat>mLat)&(nlon<MLon)&(nlon>mLon):
            if (m1[0]<MLat)&(m1[0]>mLat)&(m1[1]<MLon)&(m1[1]>mLon):
              #D=np.append(D,[[nlat, nlon,nang,'Tbh':Tbh[Band][y,x]/100,'Tbv[Band][y,x],x,y]],axis=0)
              X, Y = utm.from_latlon(nlat, nlon,force_zone_number=21)[0:2]
              Dict_Ell={'lat':nlat, 'lon':nlon,'tilt':nang,'Tbh':Tbh[Band][y,x]/100,'Tbv':Tbv[Band][y,x]/100,'AlongScan_Id':x+xmin,'AlongTrack_Id':y+ymin, 'X':X, 'Y':Y }
              Dict_Ells[n_ell]=Dict_Ell
              n_ell=n_ell+1
              
        Dict_Band['n_ell']=n_ell
        Dict_Band['filename']=filename
        Dict_Band['pass']=PASS
        Dict_Band['Ell']=Dict_Ells
        L_Kernels.Compute_Elliptic_Kernel(Dict_Band,WA_Cords) ### NEW
        Dict_Bands[Band]=Dict_Band
        #print ('Done.',end=" ")
        #sys.stdout.flush()
    Dict_Bands['Bands']=Bands
    Dict_Bands['WA_Cords']=WA_Cords
    Dict_Bands['FileName']=filename
    print("Done reading.")
    return Dict_Bands





#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   AMSR-2 L1R product reader                                               %#
#%                                                                           %#
##%############################################################################
def Read_HDF_AMSR2_L1R(path,filename,Bands,WA_Cords):#OJO L1R!!!
    Bands=[x.upper() for x in Bands]
    mLat=WA_Cords['mLat']
    mLon=WA_Cords['mLon']
    MLat=WA_Cords['MLat']
    MLon=WA_Cords['MLon']
    if ((filename.find('D_')*filename.find('A_'))>=0):
        exit()
    elif filename.find('D_')>0:
        PASS='D'
    else:
        PASS='A'
    inf=path+filename+'.h5'
    print (inf)
    infile = hdf.File(inf, "r")

    # Get the data
    Lat89A = infile["/Latitude of Observation Point for 89A"][:][:]
    Lon89A = infile["/Longitude of Observation Point for 89A"][:][:]
    
    print ("-----------------------------------------------------")
    print ("Reading HDF:"+filename)
    print ("Pass:", PASS)

    #Half power diameter for each band
    HPDMs={'C':62000,'X':42000,'KU':22000,'K':26000,'KA':12000}   #major diam in m
    HPDms={'C':35000,'X':24000,'KU':14000,'K':15000,'KA':7000}    #minor diam in m
    #A1 #6G-1.16934, #7G-0.86160, #10G-1.04596, #18G-1.08919, #23G-1.08342, #36G-0.80741
    #A2 #6G--0.03576, #7G--0.04742, #10G--0.20515, #18G-0.01587, #23G--0.06023, #36G-0.05469
    A1s={'C':0.86160,'X':1.04596,'KU':1.08919,'K':1.08342,'KA':0.80741}
    A2s={'C':-0.04742,'X':-0.20515,'KU':0.01587,'K':-0.06023,'KA':0.05469}
    
    D=Lat89A.min(axis=1)
    W=np.where((D>mLat)&(D<MLat))[0]
    y1minLat=W.min()
    y1maxLat=W.max()
    
    D=Lat89A.max(axis=1)
    W=np.where((D>mLat)&(D<MLat))[0]
    y2minLat=W.min()
    y2maxLat=W.max()
    
    ymin=min(y1minLat,y2minLat)
    ymax=max(y1maxLat,y2maxLat)+1
    
    D=Lon89A[ymin:ymax].min(axis=0)
    W=np.where((D>mLon)&(D<MLon))[0]
    x1minLon=W.min()
    x1maxLon=W.max()
    
    D=Lon89A[ymin:ymax].max(axis=0)
    W=np.where((D>mLon)&(D<MLon))[0]
    x2minLon=int(W.min()/2)*2     #so its always even
    x2maxLon=int(W.max()/2)*2+1   #so its always odd
    
    xmin=min(x1minLon,x2minLon)
    xmax=max(x1maxLon,x2maxLon)+1
    
    Lat89A =  Lat89A[ymin:ymax,xmin:xmax]
    Lon89A =  Lon89A[ymin:ymax,xmin:xmax]
    Tbh = dict()
    Tbv = dict()

    #Get data  VER SI ESTÁN BIEN LOS NOMBRES DE LAS BANDAS!!! CREO QUE VA A HACER LÍO CON EL TAMAÑO DE LAS ELIPSES QUE LEVANTA A PARTIR DEL NOMBRE DE LA BANDA, NO PUDE ENCONTRAR DONDE GUARDARLO COMO NOMBRE.

    Tbh['KA'] = infile["/Brightness Temperature (res36,36.5GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['KA'] = infile["/Brightness Temperature (res36,36.5GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbh['K']  = infile["/Brightness Temperature (res23,23.8GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbv['K']  = infile["/Brightness Temperature (res23,23.8GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbh['ku'] = infile["/Brightness Temperature (res23,18.7GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbv['ku'] = infile["/Brightness Temperature (res23,18.7GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbh['x']  = infile["/Brightness Temperature (res10,10.7GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbv['x']  = infile["/Brightness Temperature (res10,10.7GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbh['xka']= infile["/Brightness Temperature (res10,36.5GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbv['x']  = infile["/Brightness Temperature (res10,36.5GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbh['c']  = infile["/Brightness Temperature (res06,6.9GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbv['c']  = infile["/Brightness Temperature (res06,6.9GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbh['cx'] = infile["/Brightness Temperature (res06,10.7GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    #Tbv['cx'] = infile["/Brightness Temperature (res06,10.7GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbh['C']= infile["/Brightness Temperature (res06,36.5GHz,H)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Tbv['C']= infile["/Brightness Temperature (res06,36.5GHz,V)"][(ymin):(ymax),int(xmin/2):int(xmax/2)]
    Nang      = infile["/Earth Azimuth"][(ymin):(ymax),int(xmin/2):int(xmax/2)]



    Dict_Bands={}
    Dict_Bands['pass']=PASS
    for i in range(len(Bands)):
        #outWM=path+'Dict/'+infilenm+'_'+Band #working area
        print ("  Reading band:"+Band+',',  end=" ")
        sys.stdout.flush()
        
        n_ell=0
        Dict_Band={}
        Dict_Ells={}
        Dict_Band['Band']=Band
        Dict_Band['HPDm']=HPDms[Band]
        Dict_Band['HPDM']=HPDMs[Band]
        #E=np.zeros([0,5])
        for y in range(Lat89A.shape[0]):
          for x in range(int(Lat89A.shape[1]/2)):
            #Write ellipses info to DICT
            m1 = np.array([ Lat89A[y,2*x],Lon89A[y,2*x]]) 
            m2 = np.array([ Lat89A[y,2*x+1],Lon89A[y,2*x+1]])
            #print m1,m2,
            P1=(np.cos(m1[1]*np.pi/180)*np.cos(m1[0]*np.pi/180),np.sin(m1[1]*np.pi/180)*np.cos(m1[0]*np.pi/180),np.sin(m1[0]*np.pi/180))
            P2=(np.cos(m2[1]*np.pi/180)*np.cos(m2[0]*np.pi/180),np.sin(m2[1]*np.pi/180)*np.cos(m2[0]*np.pi/180),np.sin(m2[0]*np.pi/180))
            ex=np.array(P1)
            C=np.cross(ex,np.array(P2))
            ez=C/la.norm(C)
            ey=np.cross(ez,ex)
            th=np.arccos(np.dot(P1,P2))
        
            
            A1=A1s[Band]
            A2=A2s[Band]
            nP= np.cos(A2*th)*np.cos(A1*th)*ex+np.cos(A2*th)*np.sin(A1*th)*ey+np.sin(A2*th)*ez
            nlat=np.arcsin(nP[2])
            nlon=180/np.pi*np.arcsin(nP[1]/np.cos(nlat))
            nlat=180/np.pi*nlat
            
            LocalNorth=np.dot(np.array([0,0,1]),ey)*ey+np.dot(np.array([0,0,1]),ez)*ez
            LocalNorth/=la.norm(LocalNorth)
            LocalWest=np.cross(ex,LocalNorth)

            #####RAFA!!para el tilt usamos el earth azimut o dejamos el cálculo? ver en lector de AMSR-E
            nang=180/np.pi*np.arccos(np.dot(LocalNorth,ez))*np.sign(np.dot(LocalWest,ez))
            
            


            #if (nlat<MLat)&(nlat>mLat)&(nlon<MLon)&(nlon>mLon):
            if (m1[0]<MLat)&(m1[0]>mLat)&(m1[1]<MLon)&(m1[1]>mLon):
              #D=np.append(D,[[nlat, nlon,nang,'Tbh':Tbh[Band][y,x]/100,'Tbv[Band][y,x],x,y]],axis=0)
              X, Y = utm.from_latlon(nlat, nlon,force_zone_number=21)[0:2]
              Dict_Ell={'lat':nlat, 'lon':nlon,'tilt':Nang[y,x]/100,'Tbh':Tbh[Band][y,x]/100,'Tbv':Tbv[Band][y,x]/100,'AlongScan_Id':x+xmin,'AlongTrack_Id':y+ymin, 'X':X, 'Y':Y }
              Dict_Ells[n_ell]=Dict_Ell
              n_ell=n_ell+1
              
        Dict_Band['n_ell']=n_ell
        Dict_Band['filename']=filename
        Dict_Band['pass']=PASS
        Dict_Band['Ell']=Dict_Ells
        L_Kernels.Compute_Elliptic_Kernel(Dict_Band,WA_Cords) ### NEW
        Dict_Bands[Band]=Dict_Band
        print ('100%',end=" ")
        sys.stdout.flush()
    Dict_Bands['Bands']=Bands
    Dict_Bands['WA_Cords']=WA_Cords
    Dict_Bands['FileName']=filename
    return Dict_Bands


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%   AMSR-E L2A product reader                                               %#
#%                                                                           %#
##%############################################################################
def Read_HDF_AMSR_E_L2A(path,filename,Bands,WA_Cords):
    Bands=[x.upper() for x in Bands]
    mLat=WA_Cords['mLat']
    mLon=WA_Cords['mLon']
    MLat=WA_Cords['MLat']
    MLon=WA_Cords['MLon']
    if ((filename.find('_D')*filename.find('_A'))>0):
        exit()
    elif filename.find('_D')>=0:
        PASS='D'
    else:
        PASS='A'
    inf=path+filename+'.hdf'
    print (inf)
    #uncomment to read AMSRE files
    #hdf = SD(inf, SDC.READ)
    #print (hdf.datasets())

    # Get the data
    Lat89A = hdf.select("Latitude")[:][:]
    Lon89A = hdf.select("Longitude")[:][:]
    
    print ("-----------------------------------------------------")
    print ("Reading HDF:"+filename)
    print ("Pass:", PASS)

    #Half power diameter for each band - amsr-2
    #HPDMs={'c':62000,'x':42000,'ku':22000,'K':26000,'ka':12000}   #major diam in m
    #HPDms={'c':35000,'x':24000,'ku':14000,'K':15000,'ka':7000}    #minor diam in m

    #amsr-e
    HPDMs={'C':75400,'X':51400,'KU':27400,'K':31500,'KA':14400}  #major diam in m
    HPDms={'C':43200,'X':29400,'KA':15700,'K':18100,'KA': 8200}  #minor diam in m
   
    
    #AMSR-2
    #A1 #6G-1.16934, #7G-0.86160, #10G-1.04596, #18G-1.08919, #23G-1.08342, #36G-0.80741
    #A2 #6G--0.03576, #7G--0.04742, #10G--0.20515, #18G-0.01587, #23G--0.06023, #36G-0.05469
    #A1s={'c':0.86160,'x':1.04596,'ku':1.08919,'K':1.08342,'ka':0.80741}
    #A2s={'c':-0.04742,'x':-0.20515,'ku':0.01587,'K':-0.06023,'ka':0.05469}

    #amsr-e - http://nsidc.org/data/docs/daac/amsrel1a_raw_counts/header.html#psa actualizado 2011
    #A1s={'c':1.15500,'x':0.85700,'ku':0.81800,'K':0.80800,'ka':0.72200}
    #A2s={'c':0.67800,'x':0.42900,'ku':0.03100,'K':0.18500,'ka':0.06900}
 
    
    
    D=Lat89A.min(axis=1)
    W=np.where((D>mLat)&(D<MLat))[0]
    y1minLat=W.min()
    y1maxLat=W.max()
    
    D=Lat89A.max(axis=1)
    W=np.where((D>mLat)&(D<MLat))[0]
    y2minLat=W.min()
    y2maxLat=W.max()
    
    ymin=min(y1minLat,y2minLat)
    ymax=max(y1maxLat,y2maxLat)+1
    
    D=Lon89A[ymin:ymax].min(axis=0)
    W=np.where((D>mLon)&(D<MLon))[0]
    x1minLon=W.min()
    x1maxLon=W.max()
    
    D=Lon89A[ymin:ymax].max(axis=0)
    W=np.where((D>mLon)&(D<MLon))[0]
    x2minLon=int(W.min()/2)*2     #so its always even
    x2maxLon=int(W.max()/2)*2+1   #so its always odd
    
    xmin=min(x1minLon,x2minLon)
    xmax=max(x1maxLon,x2maxLon)+1
    
    Lat89A =  Lat89A[ymin:ymax,xmin:xmax]
    Lon89A =  Lon89A[ymin:ymax,xmin:xmax]
    Tbh = dict()
    Tbv = dict()

    #Get data
	 
    Tbh['KA']  =  hdf.select("36.5H_Res.4_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbv['KA']  =  hdf.select("36.5V_Res.4_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbh['K']   =  hdf.select("23.8H_Approx._Res.3_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbv['K']   =  hdf.select("23.8V_Approx._Res.3_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbh['KU']  =  hdf.select("18.7H_Res.3_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbv['KU']  =  hdf.select("18.7V_Res.3_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbh['X']   =  hdf.select("10.7H_Res.2_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbv['X']   =  hdf.select("10.7V_Res.2_TB_(not-resampled)")[:,:][(ymin):(ymax),int(xmin):int(xmax)]
    Tbh['C']   =  hdf.select("6.9H_Res.1_TB_(not-resampled)")[:,:] [(ymin):(ymax),int(xmin):int(xmax)]
    Tbv['C']   =  hdf.select("6.9V_Res.1_TB_(not-resampled)")[:,:] [(ymin):(ymax),int(xmin):int(xmax)]
    Nang= hdf.select("Earth_Azimuth")[:,:][(ymin):(ymax),int(xmin):int(xmax)]

    Dict_Bands={}
    Dict_Bands['pass']=PASS
    for i in range(len(Bands)):
        #outWM=path+'Dict/'+infilenm+'_'+Band #working area
        print ("  Reading band:"+Band+',',  end=" ")
        sys.stdout.flush()
        
        n_ell=0
        Dict_Band={}
        Dict_Ells={}
        Dict_Band['Band']=Band
        Dict_Band['HPDm']=HPDms[Band]
        Dict_Band['HPDM']=HPDMs[Band]
        #E=np.zeros([0,5])
        for y in range(Lat89A.shape[0]):
          for x in range(int(Lat89A.shape[1])):
            #Write ellipses info to DICT
            #m1 = np.array([ Lat89A[y,x],Lon89A[y,x]]) 
            nlat=Lat89A[y,x]
            nlon=Lon89A[y,x]
	    

            #if (nlat<MLat)&(nlat>mLat)&(nlon<MLon)&(nlon>mLon):
            if (nlat<MLat)&(nlat>mLat)&(nlon<MLon)&(nlon>mLon):
              #D=np.append(D,[[nlat, nlon,nang,'Tbh':Tbh[Band][y,x]/100,'Tbv[Band][y,x],x,y]],axis=0)
              X, Y = utm.from_latlon(nlat, nlon,force_zone_number=21)[0:2]


              ###Scale factor 0.1, el offset value de +327.68
              Dict_Ell={'lat':nlat, 'lon':nlon,'tilt':Nang[y,x]/100,'Tbh':Tbh[Band][y,x]/100+327.68,'Tbv':Tbv[Band][y,x]/100+327.68,'AlongScan_Id':x+xmin,'AlongTrack_Id':y+ymin, 'X':X, 'Y':Y }
              
              Dict_Ells[n_ell]=Dict_Ell
              n_ell=n_ell+1
              
        Dict_Band['n_ell']=n_ell
        Dict_Band['filename']=filename
        Dict_Band['pass']=PASS
        Dict_Band['Ell']=Dict_Ells
        L_Kernels.Compute_Elliptic_Kernel(Dict_Band,WA_Cords) ### NEW
        Dict_Bands[Band]=Dict_Band
        print ('100%',end=" ")
        sys.stdout.flush()
    Dict_Bands['Bands']=Bands
    Dict_Bands['WA_Cords']=WA_Cords
    Dict_Bands['FileName']=filename
    return Dict_Bands
