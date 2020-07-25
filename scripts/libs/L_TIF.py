# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:00:34 2017

@author: rgrimson
"""
from __future__ import division
import gdal
from osgeo import gdal
from osgeo import osr
import numpy as np
import os, sys



#%% Raster General Functions
def loadStack(src_filename):
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    tifArray = dataset.ReadAsArray()
    return tifArray

def loadStackAndGeoTransform(src_filename):
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    tifArray = dataset.ReadAsArray()
    return tifArray, dataset.GetGeoTransform()
    
def Save_Stk_GTIFF(src_filename, dst_filename,data):
    format = "GTIFF"
    driver = gdal.GetDriverByName( format )
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    if len(data.shape)>2:
        for i in range(data.shape[2]):
            dst_ds.GetRasterBand(i+1).WriteArray(data[:,:,i])
    else: #only one band
        dst_ds.GetRasterBand(1).WriteArray(data[:,:])
    dst_ds  = None
  

#save image in GeoTiff format. Use geocoding from src image
def Save_GTIFF(src_filename, dst_filename,data):
    format = "GTIFF"
    driver = gdal.GetDriverByName( format )
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds  = None

def Disp(Img,vmin=0,vmax=0,fname=""): 
    from scipy import percentile
    import matplotlib.pyplot as plt


    if (vmin==vmax):
        vmin=percentile(Img,2)
        vmax=percentile(Img,98)
    plt.imshow(Img,vmin=vmin,vmax=vmax,cmap = plt.get_cmap('gray'),interpolation='None')
    plt.axis('off')
    if (fname!=""):
        plt.savefig(fname,bbox_inches='tight')
        
def Save_Stk_GTIFF_byVRT(path,in_fn,out_fn,field,X1,Y1,X2,Y2,dx,dy,nlt,LTP):
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
def Create_Stk_GTIFF_Coords(path,out_fn,Context,M):
    image_size = np.array(list(M.shape))[[1,2]]
    l=M.shape[0]
    
    #  Choose some Geographic Transform (Around Lake Tahoe)
    lon=[Context['WA_Cords']['X1'],Context['WA_Cords']['X2']]
    lat=[Context['WA_Cords']['Y1'],Context['WA_Cords']['Y2']]
    
    #  Set the Pixel Data (Create some boxes)
    
    # set geotransform
    nx = image_size[0]
    ny = image_size[1]
    xmin, ymin, xmax, ymax = [min(lon), min(lat), max(lon), max(lat)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    
    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(path+out_fn+'.tif', int(ny), int(nx), int(l), gdal.GDT_Float32)
    
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    srs.ImportFromEPSG(32721)                # WGS84 UTM 21S mts
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    for i in range(l):
        dst_ds.GetRasterBand(i+1).WriteArray(M[i])   # write r-band to the raster
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None


#Create_Stk_GTIFF(path,out_fn,Context,M)