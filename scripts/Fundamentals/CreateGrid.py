# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:01:34 2017

@author: rgrimson

To create a new grid:

1) define a working directory and create the file my_base_dir.py with the content:
   def get():
      return base_dir
2) create a sub dir base_dir/Mapas/
3) define the new grid name: grid and greate a subdir base_dir/Mapas/grid
4) set the variable MAPA bellow to this new grid name
5) create the subdirs "HDF" and put the .h5 files, 4 empty dirs: "GC", "Imgs", "Salidas" and "Out_Grids"
6) Create another subdir: "Grids" and copy inside a shapefile (with the variable bellow ltype_shp_fn equal to this filename) containing the differnt regions in your grid
7) create files "param.txt" and "Bands.txt" in the grid dir containing 

"""

from __future__ import print_function
from __future__ import division
import numpy as np

import os
import ogr
import shutil

from shapely.geometry import shape, mapping
import fiona
import sys
from collections import Counter


#Constants
ltype_shp_fn='Land_types'
grid_shp_fn='WA_SqGrid'

#WDIR
import my_base_dir
BASE_DIR = my_base_dir.get()
#MAPA='Delta_1000km2'
MAPA='Reg25'
#MAPA='LCA_25000m'
#MAPA='LCA_50000m'
wdir=BASE_DIR + 'Mapas/%s/'%MAPA



import L_TIF
import L_SHP
import L_Context

#%%
def Dissolve_Small_cells(in_fn,cell_size,merge=True):
    #open regular grid intersectid with land types.    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(in_fn+'.shp', 1)
    layer = dataSource.GetLayer()
    Failed=False
    
    #create dict D with layer features    
    D={}
    mx=0
    my=0
    mlt=0
    for feature in layer:
        Id=feature.GetField('id')
        X=feature.GetField('X')
        Y=feature.GetField('Y')
        A=feature.GetField('Area')
        LT=feature.GetField('Land_type')
        D[Id]={'X':X,'Y':Y, 'Area':A,'Land_type':LT,'Coords':'[%d, %d]'%(X,Y)}
        if X>mx: mx = X 
        if Y>my: my = Y
        if LT>mlt: mlt = LT
    mx=mx+1
    my=my+1     
    mlt=mlt+1

    #Create new NewId matrix.
    #Starts with original Id, (-1) for points not corresponding to cells
    NewId=-1*np.ones([mx,my,mlt],dtype=int)
    for Id in D:
        X=D[Id]['X']
        Y=D[Id]['Y']
        LT=D[Id]['Land_type']
        NewId[X,Y,LT]=Id
    
    #join regions following an increasing order
    if merge:
        skip=0
        O=sorted(D,key=lambda Id: D[Id]['Area'])
        Id=O[0]
        cell=D[Id]
        A=cell['Area']
        while (A<(0.8*cell_size)):
            X=cell['X']
            Y=cell['Y']
            LT=cell['Land_type']
            
            #Find sorrounding cells        
            Xp=X+1
            Xm=X-1
            if Xp==mx: Xp=X
            if Xm==-1: Xm=0    
            Yp=Y+1
            Ym=Y-1
            if Yp==my: Yp=Y
            if Ym==-1: Ym=0    
    
            #find the smallest sorrounnding cell
            error=4*cell_size
            bx=X
            by=Y
            for x,y in [[Xm,Y],[Xp,Y],[X,Ym],[X,Yp]]:
                nid=NewId[x,y,LT]
                if (nid>=0):
                  A=D[nid]['Area']
                  if ((X!=x) or (Y!=y)) and((A<error)and(A>0)) and (nid!=Id): #a real better position (not the same, not null)
                    bx=x
                    by=y
                    error=A
            if (not ((bx==X) and (by==Y))):#found best to join?
                #join it with old Id
                nid=NewId[bx,by,LT]
                print (Id, "->", nid, '(',X, Y, ') -> (',bx, by, ') LT:', LT)
                if (nid>=0):
                    NewId[np.where(NewId==Id)]=nid
                    D[nid]['Area']+=D[Id]['Area']
                    D[nid]['Coords']+='<--'+D[Id]['Coords']
                    D.pop(Id)
                #print Id, "->", nid
            else:
                #no la puedo unir a nadie
                print ("***"            )
                print ("****************************************************")
                print ("Error, cannot find a cell to merge cell", X, Y, LT)
                Failed=True
                skip+=1
            O=sorted(D,key=lambda Id: D[Id]['Area'])
            Id=O[skip]
            cell=D[Id]
            A=cell['Area']
                            
    
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(in_fn+'.shp', 1)
        layer = dataSource.GetLayer()
        field_name = ogr.FieldDefn("Coords", ogr.OFTString)
        field_name.SetWidth(48)
        layer.CreateField(field_name)
    
        #DissolvedCells={}
        
        for feature in layer:
            X=feature.GetField('X')
            Y=feature.GetField('Y')
            LT=feature.GetField('Land_type')
            #
            nid=NewId[X,Y,LT]
            feature.SetField("Id", int(nid))
            layer.SetFeature(feature)
            #
            feature.SetField("Area", D[nid]['Area'])
            layer.SetFeature(feature)
            
            feature.SetField("Coords", D[nid]['Coords'])
            layer.SetFeature(feature)
    
    
            #        if ((np.array(DissolvedCells.keys())==nid).sum()==0):
            #            DissolvedCells[nid]='[%d, %d]'%(X,Y)
            #        else:
            #            DissolvedCells[nid]=DissolvedCells[nid] + ',[%d, %d]'%(X,Y)
            #        feature.SetField("DissolvedCells", DissolvedCells[nid])
            #        layer.SetFeature(feature)
            #


    dataSource = None

    #DISSOLVE GRID into CELLS
    L_SHP.Dissolve(in_fn+'.shp',in_fn+'_.shp','Id')
    
    #Renumber Id to get consequtive Ids
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(in_fn+'_.shp', 1)
    layer = dataSource.GetLayer()
    nid=0
    for feature in layer:
        feature.SetField("Id", int(nid))
        layer.SetFeature(feature)
        nid+=1
        #
    dataSource = None
    if Failed:
        print ("Warning, some cells (%d) could not be completely merged" %skip)
    

#%%    
def Rasterize_Land_type(path,in_fn,out_fn,field,X1,Y1,X2,Y2,dx,dy):
    os.system("gdal_rasterize -a %(field)s -l %(layer)s -te %(X1)d %(Y1)d %(X2)d %(Y2)d -tr %(dx)d %(dy)d  \"%(path)s%(in_fn)s.shp\" \"%(path)s%(out_fn)s.tif\" -ot Float32" %{'layer':in_fn,'dx':dx,'dy':dy,'field':field,'path':path,'X1':X1,'Y1':Y1,'X2':X2,'Y2':Y2,'in_fn':in_fn,'out_fn':out_fn})

def Rasterize_Land_props(path,in_fn,out_fn,field,X1,Y1,X2,Y2,dx,dy,nlt,LTP):
    print ("  Prepearing...") #an empty tmp.vrt -> tmp.tif, and then save data in that same file.
    Rasterize_Land_type(path,in_fn,'tmp',field,X1,Y1,X2,Y2,dx,dy)
    s=''    
    for i in range(nlt):
        s=s+path+'tmp.tif '
    print ("  Creating Stack")
    os.system('gdalbuildvrt -separate '+path+'tmp.vrt '+s)
    print ("  Converting to GeoTIFF")
    os.system('gdal_translate '+path+'tmp.vrt %(out_fn)s.tif' %{'out_fn':path+out_fn})
    L_TIF.Save_Stk_GTIFF(path+out_fn+'.tif', path+out_fn+'.tif',LTP)

    for fname in [path+'tmp.vrt',path+'tmp.tif']:
        try:
          os.remove(fname)
        except OSError:
          pass
    
#%%
def Create_Basic_SqGrid(fpath,X1,Y1,X2,Y2,dx,dy):
    #create fishnets
    global grid_shp_fn,ltype_shp_fn
    L_SHP.remove_shp(fpath+grid_shp_fn)
    WA_ny,WA_nx=L_SHP.fishnet(fpath+grid_shp_fn+'.shp', X1,X2,Y1,Y2,dy,dx,0,0)
    shutil.copy(fpath+ltype_shp_fn+'.prj', fpath+grid_shp_fn+'.prj') #a file named utm21s.prj must exist (or utm zone instead of 21s) Changed +'UTM'+utmzone+ for +Land_types+


#%% Particular Important Functions


#%%
def Grillar(WA_Cords,fpath,merge=True):

    global grid_shp_fn
    global ltype_shp_fn#='Land_types'
    out_grid_shp_fn='WA_Grid'
    LTSQC='Land_types_SQ_Coarse'
    LTSQF='Land_types_SQ_Fine'
    
    X1=WA_Cords['X1']
    Y1=WA_Cords['Y1']
    X2=WA_Cords['X2']
    Y2=WA_Cords['Y2']
    DX=WA_Cords['DX']
    DY=WA_Cords['DY']
    dx=DX/100
    dy=DY/100
    
    L_SHP.remove_shp(fpath+grid_shp_fn)
   
    #create fishnets
    WA_ny,WA_nx=L_SHP.fishnet( fpath+grid_shp_fn+'.shp', X1,X2,Y1,Y2,DY,DX,0,0)
    shutil.copy(fpath+ltype_shp_fn+'.prj', fpath+grid_shp_fn+'.prj') #a file named utm21s.prj must exist (or utm zone instead of 21s) Changed +'UTM'+utmzone+ for +Land_types+
    shutil.copy(fpath+ltype_shp_fn+'.prj', fpath+out_grid_shp_fn+'.prj') #a file named utm21s.prj must exist (or utm zone instead of 21s) Changed +'UTM'+utmzone+ for +Land_types+
    
    #create temporary WA grid to detect candidates to multiple-land_types cells
    Rasterize_Land_type(fpath,ltype_shp_fn,LTSQC,'Land_type',X1,Y1,X2,Y2,DX,DY)
    LT=L_TIF.loadStack(fpath+LTSQC+'.tif')
    for fname in [fpath+LTSQC+'.tif']:
        try:
          os.remove(fname)
        except OSError:
          pass
    #create temporary FINE WA grid to detect candidates to multiple-land_types cells
    Rasterize_Land_type(fpath,ltype_shp_fn,LTSQF,'Land_type',X1,Y1,X2,Y2,dx,dy)
    LTF=L_TIF.loadStack(fpath+LTSQF+'.tif')
    for fname in [fpath+LTSQF+'.tif']:
        try:
          os.remove(fname)
        except OSError:
          pass
    
    B=np.zeros(LT.shape,dtype='int')
    nB=0
    nBi=0

    nx=LT.shape[0]
    ny=LT.shape[1]

    #search for changes in the cell
    mx=int(DX/dx)
    my=int(DY/dy)
    for i in range(nx):
        for j in range(ny):
            lt=LT[i,j]
            for u in range(mx):
                for v in range(my):
                    if (LTF[u+i*mx,v+j*my]!=lt):
                        if B[i,j]!=1:
                            B[i,j]=1
                            nB+=1
    
    #search for changes in the sorrounding cells
    for i in range(nx):
        mx=max(0,i-1)
        Mx=min(i+1,nx-1)
        for j in range(ny):
            my=max(0,j-1)
            My=min(j+1,ny-1)
            lt=LT[i,j]
            for u in np.arange(mx,Mx+1):
                for v in np.arange(my,My+1):
                    if (LT[u,v]!=lt):
                        if B[i,j]!=1:
                            B[i,j]=1
                            nB+=1
    
    #open WA grid to intersect with ltypes
    grid_shp = fiona.open(fpath+grid_shp_fn+".shp")
    lt_shp = fiona.open(fpath+ltype_shp_fn+".shp") 
    schema = grid_shp.schema.copy()
    schema['properties'][u'Land_type']='int:10'
    schema['properties'][u'Area']='int:10'
    schema['properties'][u'Id']='int:10'
    Id_i=0
    print ('Grilling (computing intersections for fishnet and land_types): Total',nB,'intersections')
    sys.stdout.flush()
    #schema = {'geometry': 'Polygon','properties': {'test': 'int'}}

    #compute intersection
    with fiona.open(fpath+out_grid_shp_fn+'.shp','w','ESRI Shapefile', schema) as e:
         for cell in list(grid_shp):
           cell['properties'].update({u'Land_type':'int:10'})
           props=cell['properties']  
           X=props['X']
           Y=props['Y']
           lt=int(LT[Y,X])
           if (B[Y,X]==0):
               props['Land_type']=lt
               props['Area']=shape(cell['geometry']).area
               props['Id']=Id_i
               Id_i+=1
               geom=shape(cell['geometry'])
               e.write({'geometry':mapping(geom), 'properties':props})                
           else: #it might be a border cell
             for lt_poly in list(lt_shp):
               for geom in [shape(cell['geometry']).intersection(shape(lt_poly['geometry']))]:
                       if not geom.is_empty:
                           lt=lt_poly['properties']['Land_type']
                           props['Land_type']=lt
                           props['Area']=geom.area
                           props['Id']=Id_i
                           Id_i+=1
                           nBi+=1
                           #print props
                           print (nBi,end=" ")
                           e.write({'geometry':mapping(geom), 'properties':props})
                           #if int(nBi*Sc/nB)>pct: #print ad    ance
                           # pct=int(i*Sc/nB)
                           # print pct*Sc, "%",
                           sys.stdout.flush()


    print ('Merging Small Cells')
    Dissolve_Small_cells(fpath+out_grid_shp_fn,DX*DY,merge)    
    L_SHP.rename(fpath+out_grid_shp_fn+'_',fpath+out_grid_shp_fn)
    L_SHP.remove_shp(fpath+grid_shp_fn)
#%%
def Rasterize_Land_type_proportions(WA_Cords,fpath,margin=0):
    global ltype_shp_fn#='Land_types'
    out_grid_tif_fn='Land_type_Fine_Margin'
    LTSQUF='Land_types_SQ_UFine' #tif
    M=10 # each small square is divided into MxM subsquares (Ultra Fine = UF)
    
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    marginx=np.ceil(margin/dx)*dx
    marginy=np.ceil(margin/dy)*dy
    
    cols=WA_Cords['cols']+2*int(marginx/dx)
    rows=WA_Cords['rows']+2*int(marginy/dy)


    X1=WA_Cords['X1']-marginx
    Y1=WA_Cords['Y1']-marginy
    X2=WA_Cords['X2']+marginx
    Y2=WA_Cords['Y2']+marginy
    
    
    print ("Computing Land Types")
    Rasterize_Land_type(fpath,ltype_shp_fn,LTSQUF,'Land_type',X1,Y1,X2,Y2,dx/M,dy/M)
    #Rasterize_Land_type(fpath,ltype_shp_fn,'Cell_id_SQ_UFine','Id',X1,Y1,X2,Y2,dx/M,dy/M)
    
    LTUF=L_TIF.loadStack(fpath+LTSQUF+'.tif')    
    nlt=int(LTUF.max()+1)
    LTP=np.zeros([rows,cols,nlt]) #land type proportion
    LT=np.zeros([rows,cols]) #main land_type

    print ("Computing Proportions")
    for x in range(cols):
        for y in range(rows):
            R=LTUF[y*M:(y*M+M),x*M:(x*M+M)]
            for lt in range(nlt):
                LTP[y,x,lt]=len(np.where(R==lt)[0])/M/M
            LT[y,x]=LTP[y,x,:].argmax()
    
    print ("Saving and Removing tmp files")
    Rasterize_Land_props(fpath,ltype_shp_fn,out_grid_tif_fn,'Land_type',X1,Y1,X2,Y2,dx,dy,1,LT)
    for fname in [fpath+LTSQUF+'.tif']:
        try:
          os.remove(fname)
        except OSError:
          pass
    #%% creat tiff with cell id
def Rasterize_Cell_ID(WA_Cords,fpath,margin=0):
    CellId_shp_fn='WA_Grid'
    M=10 # each small square is divided into MxM subsquares (Ultra Fine = UF)
    CidSQUF='Cell_id_SQ_UFine.tif'
    out_fn='Cell_ID_Fine'
    
    dx=WA_Cords['dx']
    dy=WA_Cords['dy']
    marginx=np.ceil(margin/dx)*dx
    marginy=np.ceil(margin/dy)*dy
    
    cols=WA_Cords['cols']+2*int(marginx/dx)
    rows=WA_Cords['rows']+2*int(marginy/dy)


    X1=WA_Cords['X1']-marginx
    Y1=WA_Cords['Y1']-marginy
    X2=WA_Cords['X2']+marginx
    Y2=WA_Cords['Y2']+marginy
    
    
    print ("Computing Cells Raster")
    Rasterize_Land_type(fpath,CellId_shp_fn,CidSQUF,'Id',X1,Y1,X2,Y2,dx/M,dy/M)
    
    LTUF=L_TIF.loadStack(fpath+CidSQUF+'.tif')    
    nv=int(LTUF.max()+1)
    #CidP=np.zeros([rows,cols,nv]) #cell id proportion
    Cid=np.zeros([rows,cols]) #dominant cell id

    print ("Computing Proportions %d"%cols)
    for x in range(cols):
        print(x,end=' ')
        sys.stdout.flush()
        for y in range(rows):
            R=LTUF[y*M:(y*M+M),x*M:(x*M+M)].astype(int).reshape(M*M)
            Cid[y,x]=Counter(R).most_common(1)[0][0]
    
    print ("Saving tmp files")
    Rasterize_Land_props(fpath,CellId_shp_fn,out_fn,'Land_type',X1,Y1,X2,Y2,dx,dy,1,Cid)
    for fname in [fpath+CidSQUF+'.tif']:
        try:
          os.remove(fname)
        except OSError:
          pass
  



#%%
if __name__ == "__main__":
    print ("Create Grids")
    os.chdir(wdir)
    Param=L_Context.read_param()
    fpath=wdir+'Grids/'
    
    WA_Cords=L_Context.Compute_Working_Area_UTM_coordinates(Param)
    Rasterize_Land_type_proportions(WA_Cords,fpath,Param['GC_margin'])
    Grillar(WA_Cords,fpath,merge=True)
    Rasterize_Cell_ID(WA_Cords,fpath,margin=0)
