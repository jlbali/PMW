# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:01:25 2017

@author: rgrimson
"""
from __future__ import print_function

from __future__ import division
import numpy as np

import os
import ogr
import shutil
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import fiona
import itertools

#%%
def Read_Fields(fn):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(fn+'.shp')
    layer = dataSource.GetLayer()
    layerDefinition = layer.GetLayerDefn()
    
    
    for i in range(layerDefinition.GetFieldCount()):
        print (layerDefinition.GetFieldDefn(i).GetName())
        

#%% Shapefile General Functions
def fishnet(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth,colSt=1,rowSt=1):
    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = np.ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = np.ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    XField = ogr.FieldDefn("X", ogr.OFTInteger)
    YField = ogr.FieldDefn("Y", ogr.OFTInteger)
    RField = ogr.FieldDefn("Land_type", ogr.OFTInteger)
    outLayer.CreateField(XField)
    outLayer.CreateField(YField)
    outLayer.CreateField(RField)

    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outFeature.SetField("X",countcols-1+colSt)
            outFeature.SetField("Y",countrows-1+rowSt)
            outFeature.SetField("Land_type",0)
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Close DataSources
    outDataSource.Destroy()
    return int(rows), int(cols)
#%%
def remove_shp(fname):
  for ext in [".shp",".prj",".shx",".dbf"]:
    try:
      os.remove(fname+ext)
    except OSError:
      pass

#%%
def rename(old,new):
    for ext in [".shp",".shx",".dbf",".prj"]:
      try:
        os.remove(new+ext)
      except:
        pass
      try:
        shutil.copy(old+ext, new+ext) 
      except:
        pass    
      try:
        os.remove(old+ext)
      except:
        pass    
#%%
def diff(fpath,A,B):
    import fiona
    from shapely.geometry import shape
    green = fiona.open(fpath+A+".shp")
    lt_shp = fiona.open(fpath+B+".shp") 
    from shapely.geometry import mapping
    schema = green.schema.copy()
    #schema = {'geometry': 'Polygon','properties': {'test': 'int'}}
    with fiona.open(fpath+'diff.shp','w','ESRI Shapefile', schema) as e:
         for i in list(green):
          for j in list(lt_shp):
             for geom in [shape(i['geometry']).difference(shape(j['geometry']))]:
                if not geom.is_empty:
                       props=i['properties']
                       lt=j['properties']['Land_type']
                       props['Land_type']=lt
                       print (props)
                       e.write({'geometry':mapping(geom), 'properties':props})
         #for geom  in [shape(i['geometry']).difference(shape(j['geometry'])) for i,j in zip(list(blue),list(green))]:
         #   if not geom.is_empty:
         #          e.write({'geometry':mapping(geom), 'properties':{'test':2}})
#%%
def Dissolve(in_fn,out_fn,groupby):
    #groupby='Land_type'#'newId'
    #in_fn=path+'Grids/WA_Grid.shp'
    #out_fn=path+'Grids/dissolve.shp'
    with fiona.open(in_fn) as input:
        # preserve the schema of the original shapefile, including the crs
        meta = input.meta
        with fiona.open(out_fn, 'w', **meta) as output:
            # groupby clusters consecutive elements of an iterable which have the same key so you must first sort the features by the 'newId' field
            e = sorted(input, key=lambda k: k['properties'][groupby])
            # group by the 'newId' field 
            for key, group in itertools.groupby(e, key=lambda x:x['properties'][groupby]):
                properties, geom = zip(*[(feature['properties'],shape(feature['geometry'])) for feature in group])
                # write the feature, computing the unary_union of the elements in the group with the properties of the first element in the group
                output.write({'geometry': mapping(unary_union(geom)), 'properties': properties[0]})

#%%                           
def Add_area_field(filename):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(filename+'.shp', 1)
    layer = dataSource.GetLayer()
    new_field = ogr.FieldDefn("Area", ogr.OFTReal)
    new_field.SetWidth(32)
    layer.CreateField(new_field)
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea() 
        feature.SetField("Area", area)
        layer.SetFeature(feature)
    
    dataSource = None


def getNeighbors(filepath):
    import fiona
    from shapely.geometry import shape
    shapefile = fiona.open(filepath + ".shp")
    neighbors = {}
    features = list(shapefile)
    for feature1 in features:
        #print "Feature1 properties: ", feature1['properties']
        print ("Calculando vecinos de ", feature1['properties']['Id'], " de un total de ", len(features))
        vecinos = []
        neighbors[feature1['properties']['Id']] = vecinos
        for feature2 in list(shapefile):
            # Salteamos si el id es el mismo.
            if feature1['properties']['Id'] == feature2['properties']['Id']:
                continue
            # Saltemos si NO son del mismo LandType.
            if feature1['properties']['Land_type'] != feature2['properties']['Land_type']:
                continue
            if shape(feature1['geometry']).touches(shape(feature2['geometry'])):
                vecinos.append(feature2['properties']['Id'])
    return neighbors

# Hace el corte usando los bounding boxes.
def getFastNeighbors(filepath):
    import fiona
    from shapely.geometry import shape, box
    shapefile = fiona.open(filepath + ".shp")
    neighbors = {}
    features = list(shapefile)
    # Precalculamos bounding boxes.
    bBoxes = {}
    print("Iniciando proceso de precomputo de Bounds...")
    for feature in features:
        Id = feature['properties']['Id']
        bounds = shape(feature['geometry']).bounds
        bbox =  box(bounds[0], bounds[1], bounds[2], bounds[3])
        bBoxes[Id] = bbox
    print("Finalizado el proceso de precomputo de bounds")
    counter = 0
    for feature1 in features:
        #print "Feature1 properties: ", feature1['properties']
        if counter % 50 == 0:
            print ("Calculando vecinos de ", feature1['properties']['Id'], " de un total de ", len(features))
        counter = counter + 1
        vecinos = []
        neighbors[feature1['properties']['Id']] = vecinos
        bbox1 = bBoxes[feature1['properties']['Id']]
        for feature2 in list(shapefile):
            bbox2 = bBoxes[feature2['properties']['Id']]
            # Salteamos si el id es el mismo.
            if feature1['properties']['Id'] == feature2['properties']['Id']:
                continue
            # Saltemos si NO son del mismo LandType.
            if feature1['properties']['Land_type'] != feature2['properties']['Land_type']:
                continue
            if bbox1.touches(bbox2):
                vecinos.append(feature2['properties']['Id'])
        #print("Cantidad de vecinos " + str(vecinos))
    return neighbors

