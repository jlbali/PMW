# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 12:39:42 2017

@author: rgrimson
"""
from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import csv
import os
import libs.L_SHP as L_SHP
import libs.L_TIF as L_TIF
import sys
import numpy as np




#%%
###############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Save CSV, SHP, TIF and JPG                                               %#
#%                                                                           %#
#%%############################################################################
def Export_Solution(Sol, Context, Band, FileName):
    if Sol==[]:
        return
    wdir=Context['wdir']
    
    SaveCSV(Sol,wdir,FileName,Band) #Creates CSV file with solution
    CSV2SHP(    wdir,FileName,Band) #Joins CSV file with Solution Grid in a SHP
    SHP2TIF(    wdir,FileName,Band) #Rasterizes SHP wo a GeoTIF file
    TIF2JPG(    wdir,FileName,Band) #Renders GeoTIF file as JPG
    print('')


def Export_Observation(Observation, Context, Band, FileName=''):
    if Observation=={}:
        return
    if FileName=='':
        FileName=Observation['FileName']        
    wdir=Context['wdir']
    os.chdir(wdir)
    
    #JPG
    S=Observation['Img']['H'].T[::-1,:].shape
    M=np.zeros([3,S[0],S[1]])
    c={'H':0,'V':1}
    for pol in ['H','V']:
        Field='Obs%s'%pol
        I=Observation['Img'][pol].T[::-1,:]
        M[c[pol]]=I
        
        
        
        jpg_file="%(out_fn)s_%(Field)s_%(Band)s.tif" %{'Field':Field,'out_fn':FileName,'Band':Band}
        new_dir="Imgs/%s/%s/" %(Field,Band)
        int_dir="Imgs/%s/" %Field
        if not os.path.exists(int_dir):
            os.makedirs(int_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        plt.clf()
        #plt.imshow(I,vmin=50,vmax=275)
        plt.imshow(I)
        plt.savefig(new_dir+jpg_file, bbox_inches='tight')
    #IP JPG
    M[2]=(M[1]-M[0])/(M[1]+M[0])
    I=M[2]
    Field='ObsIP'
    jpg_file="%(out_fn)s_%(Field)s_%(Band)s.tif" %{'Field':Field,'out_fn':FileName,'Band':Band}
    new_dir="Imgs/%s/%s/" %(Field,Band)
    int_dir="Imgs/%s/" %Field
    if not os.path.exists(int_dir):
        os.makedirs(int_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    plt.clf()
    #plt.imshow(I,vmin=.01,vmax=.2)
    #plt.savefig(new_dir+jpg_file, bbox_inches='tight')
    
    #IP Hist
    #    jpg_file="%(out_fn)s_%(Field)s_%(Band)s_HIST.tif" %{'Field':Field,'out_fn':FileName,'Band':Band}
    #    plt.clf()
    #    J=I.reshape(np.prod(I.shape))
    #        
    #    TbH=Observation['Tb']['H']
    #    TbV=Observation['Tb']['V']
    #    IP=(TbV-TbH)/(TbV+TbH)
    #    
    #    plt.hist([J,IP],bins=100,normed=True)
    #    plt.savefig(new_dir+jpg_file, bbox_inches='tight')
    
    #plt.close()    



    #GEOTIFF
    Field='ObsTIFF'
    int_dir="Out_Grids/%s/" %Field
    if not os.path.exists(int_dir):
        os.makedirs(int_dir)
    L_TIF.Create_Stk_GTIFF_Coords(int_dir,FileName+'_Obs_'+Band,Context,M)


def TIF2JPG(path,out_fn,Band): 
    print ("JPG",end=" ")
    os.chdir(path+'Out_Grids')
    
 
    Fields=['Tbh','Tbv','IP']
    map_prefix=path.split('/')[-2][4:6] #fourth and fifth chars in last dir in path
    for Field in Fields:
        tif_file="%(out_fn)s_%(Field)s_%(Band)s.tif" %{'Field':Field,'out_fn':out_fn,'Band':Band}
        #new_dir="../Imgs/Sol_%s/%s/" %(Field,Band)
        #int_dir="../Imgs/Sol_%s/" %Field
        new_dir="../Imgs/" #not 
        int_dir="../Imgs/" #not ordered
        if not os.path.exists(int_dir):
            os.makedirs(int_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        I=L_TIF.loadStack(tif_file)
        plt.clf()
        if Field in ['Tbh','Tbv']:
            L_TIF.Disp(I,vmin=150,vmax=300)
        elif Field=='IP':
            print("IP")
            I=(I<1)*I+(I>=1)*1
            I=(I>-1)*I+(I<-1)*(-1)
            ax = plt.gca()
            ax.invert_yaxis()
            #plt.pcolormesh(I, cmap=my_cmap, vmin=-1,vmax=1)
            plt.imshow(I, cmap=my_cmap, vmin=-1,vmax=1)
            plt.colorbar()
        #plt.savefig(new_dir+tif_file[:-3]+'jpg', bbox_inches='tight')
        plt.savefig(new_dir+tif_file[:-4]+'_'+map_prefix+'.jpg', bbox_inches='tight')
        plt.clf()
        #plt.close()
        
#%%
        #Export_IF(Context, Bands, FileName)
        
#%%
###############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Save CSV, SHP, TIF and JPG                                               %#
#%                                                                           %#
#%%############################################################################
def SaveCSV(Sol,path,out_fn,Band):
        print ("CSV",end=" ")
 
        n_Vars=Sol[0].shape[0]
        csv_fn = path+'Out_Grids/'+out_fn+'_'+Band+'.csv'
        names = ['Id','Tbh','Tbv','IP']
        with open(csv_fn,'w') as csvfile:
            out_csv = csv.writer(csvfile)
            out_csv.writerow(names)
            for v in range(n_Vars):
              Tbh=Sol[0][v]
              Tbv=Sol[1][v]
              IP=2*(Tbv-Tbh)/(Tbv+Tbh)
              row=[v,Tbh,Tbv,IP]
              out_csv.writerow(row)
        #print ("Done... csv")

#%%
def CSV2SHP(path,out_fn,Band):
    print ("SHP",end=" ")
    os.chdir(path+'Out_Grids')
 
    shp_fn     = 'WA_Grid'
    csv_fn     = out_fn+'_'+Band 
    joined_shp = "Joined"
    TYPE='\"Integer\"'
    RTYPE=',\"Real\"'
    
    Fields=['Tbh','Tbv','IP']
    SLCT=shp_fn+".Id as Id"
    for Field in Fields:
      SLCT=SLCT+', '+csv_fn+'.'+Field+' as '+Field
      TYPE=TYPE+RTYPE
    
    #Join
    L_SHP.remove_shp(joined_shp)
    f=open(csv_fn+'.csvt','w')
    f.write(TYPE)
    f.close() # No estaba cerrando, con lo cual no se generaba el archivo para el momento del join.
    os.system("ogr2ogr -sql \"select %(SLCT)s from %(SHP)s left join \'%(CSV)s.csv\'.%(CSV)s on %(SHP)s.Id = %(CSV)s.Id\" %(joined)s.shp ../Grids/%(SHP)s.shp" %{'CSV':csv_fn,'SHP':shp_fn,'joined':joined_shp,'SLCT':SLCT})


def CSV2SHP_test(path,out_fn,Band):
    print ("SHP",end=" ")
    os.chdir(path+'Out_Grids')
 
    shp_fn     = 'WA_Grid_copy'
    csv_fn     = out_fn+'_'+Band 
    joined_shp = "Joined"
    TYPE='\"Integer\"'
    RTYPE=',\"Real\"'
    
    Fields=['Tbh','Tbv','IP']
    SLCT=shp_fn+".Id as Id"
    SLCT=SLCT+', '+shp_fn+".Is_Border as Is_Border"    
    for Field in Fields:
      SLCT=SLCT+', '+csv_fn+'.'+Field+' as '+Field
      TYPE=TYPE+RTYPE
    
    #Join
    L_SHP.remove_shp(joined_shp)
    f=open(csv_fn+'.csvt','w')
    f.write(TYPE)
    f.close() # No estaba cerrando, con lo cual no se generaba el archivo para el momento del join.
    os.system("ogr2ogr -sql \"select %(SLCT)s from %(SHP)s left join \'%(CSV)s.csv\'.%(CSV)s on %(SHP)s.Id = %(CSV)s.Id\" %(joined)s.shp ../Grids/%(SHP)s.shp" %{'CSV':csv_fn,'SHP':shp_fn,'joined':joined_shp,'SLCT':SLCT})



def SHP2TIF(path,out_fn,Band):
    print ("TIF",end=" ")
    os.chdir(path+'Out_Grids')
    joined_shp = "Joined"#out_fn+'_'+Band+'_Joined'
    
    Res_X=1280 #should be computed, not constants
    Res_Y=1280

    Fields=['Tbh','Tbv','IP']
    for Field in Fields:
        #print ("gdal_rasterize -a %(Field)s -l %(joined_shp)s -ts %(X)d %(Y)d %(joined_shp)s.shp %(out_fn)s_%(Field)s_%(Band)s.tif -ot Float32" %{'Field':Field,'joined_shp':joined_shp,'out_fn':out_fn,'Band':Band,'X':Res_X,'Y':Res_Y})
        os.system("gdal_rasterize -a %(Field)s -l %(joined_shp)s -ts %(X)d %(Y)d %(joined_shp)s.shp %(out_fn)s_%(Field)s_%(Band)s.tif -ot Float32" %{'Field':Field,'joined_shp':joined_shp,'out_fn':out_fn,'Band':Band,'X':Res_X,'Y':Res_Y})

def CSV2SHP_borders(path,out_fn,Band):
    print ("SHP",end=" ")
    os.chdir(path+'Out_Grids')
 
    shp_fn     = 'WA_Grid'
    csv_fn     = out_fn+'_'+Band 
    joined_shp = "Joined"
    TYPE='\"Integer\"'
    RTYPE=',\"Real\"'
    
    Fields=['Tbh','Tbv','IP']
    SLCT=shp_fn+".Id as Id"
    SLCT=SLCT+', '+shp_fn+".Is_Border as Is_Border"    
    for Field in Fields:
      SLCT=SLCT+', '+csv_fn+'.'+Field+' as '+Field
      TYPE=TYPE+RTYPE
    
    #Join
    L_SHP.remove_shp(joined_shp)
    f=open(csv_fn+'.csvt','w')
    f.write(TYPE)
    f.close() # No estaba cerrando, con lo cual no se generaba el archivo para el momento del join.
    os.system("ogr2ogr -sql \"select %(SLCT)s from %(SHP)s left join \'%(CSV)s.csv\'.%(CSV)s on %(SHP)s.Id = %(CSV)s.Id\" %(joined)s.shp ../Grids/%(SHP)s.shp" %{'CSV':csv_fn,'SHP':shp_fn,'joined':joined_shp,'SLCT':SLCT})

      

def Export_Solution_tif(Sol, Context, Band, FileName):
    if Sol==[]:
        return
    wdir=Context['wdir']
    
    SaveCSV(Sol,wdir,FileName,Band) #Creates CSV file with solution
    CSV2SHP_borders(wdir,FileName,Band) #Joins CSV file with Solution Grid in a SHP
    SHP2TIF(wdir,FileName,Band) #Rasterizes SHP wo a GeoTIF file
    print('')
#%%
###############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Compute Indices and export Solutions                                     %#
#%                                                                           %#
#%%############################################################################
#%%
def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


#%%

colors = [(255,0,0), (255,255,0), (255,255,255), (0,157,0), (0,0,255)] # This example uses the 8-bit RGB
my_cmap = make_cmap(colors, bit=True)
position=[ 0., 0.25, 0.46, 0.5, 0.56,0.575,0.60,0.65,0.75,0.8,0.85,0.9,0.95,1]

colors=[(1.0,1.0,1.0), #White    -1    0
        (1.0,1.0,0.0), #Yellow   -0.5  0.25
        (0.8,0.8,0.2), #Brown    -0.1  0.45
        (0.5,0.5,0.0), #LBr      0.0  0.5
        (0.3,0.8,0.3), #Green    0.1   0.56
        (0.3,1.0,0.3), #LGreen   0.15   0.575
        (0.0,7.0,0.5), #BGreen   0.2   0.6
        (0.0,1.0,1.0), #Cyan     0.4   0.65 
        (0.5,0.5,1.0), #Cyan     0.5   0.75
        (0.3,0.3,1.0), #L Blue   0.6   0.8
        (0.2,0.2,0.5), #D Blue   0.7   0.85
        (0.0,0.0,0.2), #Black    0.8   0.9
        (0.5,0.5,0.7), #Gray     0.9   0.95
        (1.0,1.0,1.0), #White    1     1
]

#position=[0,0.001,0.01,0.02,0.04,0.1,0.2,0.25,0.3,0.37,0.44,0.5,0.6,0.7,0.8,0.9,0.99,1]
#colors=[(1.0,1.0,1.0), #White
#        (1.0,1.0,0.0), #Yellow
#        (0.5,0.7,0.0), #Yellow
#        (0.3,0.7,0.0), #Yellow
#        (0.7,0.5,0.2), #Brown
#        (0.5,0.4,0.0), #DBrown
#        (0.3,0.2,0.0), #DBrown        
#        (0.0,0.5,0.0), #Green
#        (0.0,1.0,1.0), #Cyan
#        (0.5,0.5,1.0), #Cyan
#        (0.3,0.3,1.0), #L Blue
#        (0.2,0.2,0.5), #D Blue
#        (0.0,0.0,0.2), #Black
#        (0.7,0.2,0.7), #voilet
#        (0.7,0.5,0.7), #voilet
#        (0.5,0.5,0.7), #Gray
#        (1.0,1.0,1.0), #White
#        (0.0,0.0,0.0) #White
#]
my_cmap = make_cmap(colors,position=position)

#%%
def IP_FromTif(fname):
    print("IP")
    #print ("   -IP | Loading...",end=" ")
    I=L_TIF.loadStack(fname)
    I=(I<1)*I+(I>=1)*1
    I=(I>-1)*I+(I<-1)*(-1)
    #I.dtype=float
    #print ("Rendering...",end=" ")
    plt.clf()
    ax = plt.gca()
    ax.invert_yaxis()
    
    plt.pcolormesh(I, cmap=my_cmap, vmin=-1,vmax=1)
    plt.colorbar()
    #L_TIF.Disp(I)
    #print ("Saving...")
    plt.savefig('../Imgs/'+fname[:-3]+'jpg', bbox_inches='tight')
    