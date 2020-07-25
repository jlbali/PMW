# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:03:51 2017

@author: rgrimson
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import utm
from L_TIF import loadStackAndGeoTransform
from L_TIF import loadStack
import ogr
import os

import L_SHP
import L_Files

#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Load Context (WA_Cords, Bands, Vars and Grids)                           %#
#%                                                                           %#
##%############################################################################
#%%

NO_NEIGHBORS = 0
COMPUTE_NEIGHBORS = 1
COMPUTE_FAST_NEIGHBORS = 2
LOAD_NEIGHBORS = 3
COMPUTE_IF_NECESSARY = 4

def Load_Context(wdir, ComputeNeighbors=COMPUTE_IF_NECESSARY):
    #runfile('/home/rgrimson/Dropbox/Code/PMW/Scripts/ProcessHDF.py', wdir=r'/home/rgrimson/Dropbox/Code/PMW/Scripts')
    os.chdir(wdir)
    Param=read_param()
    path=wdir
    Param['path']=wdir
    gpath=path+'Grids/'

    WA_Cords=Compute_Working_Area_UTM_coordinates(Param)
    Bands=Read_Bands_to_Process(path)
    Dict_Vars=Read_Vars(WA_Cords,gpath)
    Dict_Grids=Read_Grids(WA_Cords,gpath)
    neighbors = Read_Neighbors(gpath, ComputeNeighbors)
    Context={'wdir':wdir, 'Param':Param, 'WA_Cords':WA_Cords, 'Dict_Vars':Dict_Vars, 
        'Dict_Grids':Dict_Grids, 'Bands':Bands, 'Neighbors': neighbors, 'wdir':wdir}

    return Context

def Load_Context_NoGrid(wdir, ComputeNeighbors=COMPUTE_IF_NECESSARY):
    #runfile('/home/rgrimson/Dropbox/Code/PMW/Scripts/ProcessHDF.py', wdir=r'/home/rgrimson/Dropbox/Code/PMW/Scripts')
    os.chdir(wdir)
    Param=read_param()
    path=wdir
    Param['path']=wdir
    gpath=path+'Grids/'

    WA_Cords=Compute_Working_Area_UTM_coordinates(Param)
    Bands=Read_Bands_to_Process(path)
    Dict_Vars=Read_Vars(WA_Cords,gpath)
    Context={'wdir':wdir, 'Param':Param, 'WA_Cords':WA_Cords, 'Dict_Vars':Dict_Vars, 
        'Bands':Bands, 'wdir':wdir}

    return Context

def Read_Neighbors(gpath, ComputeNeighbors=COMPUTE_IF_NECESSARY):
    if ComputeNeighbors==COMPUTE_NEIGHBORS:
        grid_shp_fn=gpath + '/WA_Grid'
        neighbors = L_SHP.getNeighbors(grid_shp_fn)
        L_Files.save_obj(neighbors, gpath + '/neighbors')
        return neighbors
    elif ComputeNeighbors==COMPUTE_FAST_NEIGHBORS:
        grid_shp_fn=gpath + '/WA_Grid'
        neighbors = L_SHP.getFastNeighbors(grid_shp_fn)
        L_Files.save_obj(neighbors, gpath + '/neighbors')
        return neighbors        
    elif ComputeNeighbors==LOAD_NEIGHBORS:
        neighbors = L_Files.load_obj(gpath + '/neighbors')
        return neighbors
    elif ComputeNeighbors==COMPUTE_IF_NECESSARY:
        if L_Files.exists(gpath + '/neighbors.pklz'):
            return Read_Neighbors(gpath, LOAD_NEIGHBORS)
        else:
            return Read_Neighbors(gpath, COMPUTE_FAST_NEIGHBORS)


#%%############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Details                                                                  %#
#%                                                                           %#
##%############################################################################
#%%
#%%#READ GRIDS
def Read_Grids(WA_Cords,gpath):
    X1=WA_Cords['X1']
    #Y1=WA_Cords['Y1']
    #X2=WA_Cords['X2']
    Y2=WA_Cords['Y2']
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    cols=WA_Cords['cols']
    rows=WA_Cords['rows']
    
    
    Img,GeoTransform=loadStackAndGeoTransform(gpath+'Land_type_Fine_Margin.tif')
    Img=Img.astype(int)
    CIdG=loadStack(gpath+'Cell_ID_Fine.tif')[::-1,:].T
    nyI,nxI=Img.shape
    xa=GeoTransform[0]
    ya=GeoTransform[3]
    mx1=int((X1-xa)/dx)
    mx2=nxI-cols-mx1
    mx=min(mx1,mx2)
    my2=int((ya-Y2)/dy)
    my1=nyI-rows-my2
    my=min(my1,my2)
    m=min(mx,my)

    LandPropMargin_Grid=np.fliplr(Img.transpose()) #fine grid for computations. With margin for georreference.
    LandPropMargin_Grid=LandPropMargin_Grid[mx1-m:nxI-(mx2-m),my1-m:nyI-(my2-m)]
    LandType_Grid=LandPropMargin_Grid[m:m+cols, m:m+rows] ##fine land proportion grid for computations.

    Dict_Grids={}    
    Dict_Grids['LandTypeMargin_Grid']=LandPropMargin_Grid
    Dict_Grids['GeoCorr_margin']=m
    Dict_Grids['LandType_Grid']=LandType_Grid
    Dict_Grids['CellID_Grid']=CIdG.astype(int)
    nlt=int(LandType_Grid.max()+1)
    #Dict_Grids['nx']=nx
    #Dict_Grids['ny']=ny
    Dict_Grids['nlt']=nlt

    return Dict_Grids

def Read_Cell_ID_Prop_Grid(gpath):
    Img,GeoTransform=loadStackAndGeoTransform(gpath+'Cell_ID_Fine.tif')
    return Img
    
#%%#READ VARS
def Read_Vars(WA_Cords,gpath):
    grid_shp_fn='WA_Grid'

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(gpath+grid_shp_fn+'.shp')
    layer = dataSource.GetLayer()
    
    Dict_Vars={}
    Vars={}
    #mx=0
    #my=0
    mlt=0
    nv=0
    for feature in layer:#create dict D with layer features
        Id=feature.GetField('Id')
        #X=feature.GetField('X')
        #Y=feature.GetField('Y')
        A=feature.GetField('Area')
        LT=feature.GetField('Land_type')
        CoordsStr=feature.GetField('Coords')
        Coords=[list(map(int,C.replace('[','').replace(']','').split(','))) for C in CoordsStr.split('<--')]
        #Vars[Id]={'X':X,'Y':Y, 'Area':A,'Land_type':LT,'Coords':Coords}
        Vars[Id]={'Area':A,'Land_type':LT,'Coords':Coords}
        #if X>mx: mx = X 
        #if Y>my: my = Y
        if LT>mlt: mlt = LT
        nv+=1
    Dict_Vars['n_Vars']=nv
    #Dict_Vars['nx']=mx+1
    #Dict_Vars['ny']=my+1
    Dict_Vars['nlt']=mlt+1
    Dict_Vars['Var']=Vars
    VType=np.zeros(nv,dtype=int)
    for i in range(nv):
        VType[i]=Dict_Vars['Var'][i]['Land_type']
    Dict_Vars['VType']=VType
    return Dict_Vars

 
#%% READ BANDS
def Read_Bands_to_Process(path):
    fname=path+'Bands.txt'
    Bands=[]
    with open(fname) as f:
        content = f.readline().rstrip().lstrip()
        while content!='':
            Bands.append(content.rstrip().upper())
            content = f.readline().rstrip().lstrip()
    print (Bands)
    return Bands
    
    
#%% Compute_Working_Area_UTM_coordinates
def Compute_Working_Area_UTM_coordinates(Param):

    mLon = Param['mLon']
    MLon = Param['MLon']
    mLat = Param['mLat']
    MLat = Param['MLat']
    dx = Param['dx']
    dy = Param['dy']
    DX = Param['DX']
    DY = Param['DY']
    utmzone=Param['utmzone']  #utmzone='21H' #UTM zone for projection (format DDC,two digits, one char)
    margin=Param['WA_margin']

    if 'fDX' in Param.keys(): #To force a map to be compatible with a coarser one (fdx="FORCED DX" is the dx used to define map corners) fdx=k*dx for some k positive integer.
        fDX=Param['fDX']
    else:
        fDX=DX
    if 'fDY' in Param.keys():
        fDY=Param['fDY']
    else:
        fDY=DY
    nx=int(DX/dx)
    ny=int(DY/dy)
    fnx=int(fDX/dx) #To force ...
    fny=int(fDY/dy)
    
    utmzonenumber=int(utmzone[0:2])
    utmzoneletter=utmzone[2]
    
    X1, Y1 = utm.from_latlon(mLat, mLon , force_zone_number=utmzonenumber)[0:2]
    X2, Y2 = utm.from_latlon(MLat, MLon , force_zone_number=utmzonenumber)[0:2]
    
    X1 = np.floor(X1/dx)*dx - margin
    X2 = np.ceil(X2/dx)*dx  + margin
    Y1 = np.floor(Y1/dy)*dy - margin
    Y2 = np.ceil(Y2/dy)*dy  + margin

    rows = int((Y2-Y1)/dy)
    cols = int((X2-X1)/dx)
    COLS=int(np.ceil(float(cols)/nx))
    ROWS=int(np.ceil(float(rows)/ny))
    fCOLS=int(np.ceil(float(cols)/fnx)) #To force 
    fROWS=int(np.ceil(float(rows)/fny))
    
    X1=X1-((fnx*fCOLS-cols)/2)*dx
    Y1=Y1-((fny*fROWS-rows)/2)*dy

    rows=fny*fROWS
    cols=fnx*fCOLS
    X2 = X1 + dx*cols
    Y2 = Y1 + dy*rows
    #DX=dx*nx
    #DY=dy*ny
    
    MLat,mLon=utm.to_latlon(X1,Y2,utmzonenumber,utmzoneletter) #**
    mLat,MLon=utm.to_latlon(X2,Y1,utmzonenumber,utmzoneletter)
    
    WA_Cords={}
    WA_Cords['mLat']=mLat
    WA_Cords['mLon']=mLon
    WA_Cords['MLat']=MLat
    WA_Cords['MLon']=MLon
    WA_Cords['X1']=X1
    WA_Cords['Y1']=Y1
    WA_Cords['X2']=X2
    WA_Cords['Y2']=Y2
    WA_Cords['ROWS']=ROWS
    WA_Cords['COLS']=COLS
    WA_Cords['rows']=rows
    WA_Cords['cols']=cols
    WA_Cords['utmzone']=utmzone
    WA_Cords['dx']=dx
    WA_Cords['dy']=dy
    WA_Cords['DX']=DX
    WA_Cords['DY']=DY
    WA_Cords['nx']=nx
    WA_Cords['ny']=ny
    WA_Cords['ny']=ny
    
    #    WA_Cords['ncx']=int(cols/2)
    #    WA_Cords['ncy']=int(rows/2)
    #    WA_Cords['cx']=X1+dx/2+WA_Cords['ncx']*dx
    #    WA_Cords['cy']=Y1+dy/2+WA_Cords['ncy']*dy

    WA_Cords['GC_margin']=Param['GC_margin']
    WA_Cords['WA_margin']=Param['WA_margin']
    return WA_Cords

#%%
###############################################################################
#%                                                                           %#
#%                                                                           %#
#% READ PARAMETERS FILE "PARAM.TXT"                                          %#
#%                                                                           %#
#%%############################################################################
mandatory_keys_type_def={'mLon':1, 'MLon':1, 'DX':1, 'dy':1, 'MLat':1, 'utmzone':0, 'DY':1, 'dx':1, 'mLat':1, 'WA_margin':1, 'GC_margin':1}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
#%%
def read_param(in_fn='param.txt', mandatory_fields_type=mandatory_keys_type_def):
    s = open(in_fn, 'r').read().splitlines()
    D={}
    
    for l in s:
        u=l.replace('\t',' ').replace(' ','').split('=')
        if (len(u)!=2):
            if (u[0].__len__()==0):
                pass
            elif u[0][0]!='#':
                print ("Error reading param file")
                print ("in line", l)
        else:
            var=u[0].rstrip(' ')
            val=u[1].rstrip(' ')
            if is_number(val):
                D[var]=float(val)
            else:
                D[var]=val
    
    failed=False
    print ("Input parameters:")
    for k in mandatory_fields_type.keys():
        try:
           print (k,'=',D[k])
           if ((mandatory_fields_type[k]==1) and(type(D[k])!=float)):
               print ("Error, this value should be a number!")
               failed=True
           #break
        except:
           print ("Undefined!")
           failed=True
   
    if (failed):
        print ("#####################################################")
        print ("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")
        print ("#####################################################")
    return D
