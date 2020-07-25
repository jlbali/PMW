# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:29:59 2017

@author: rgrimson
"""
import pickle
import gzip
#import json
#from pathlib import Path
from os import listdir
import os.path

###############################################################################
#%                                                                           %#
#%                                                                           %#
#%  Basic Files I/O                                                          %#
#%                                                                           %#
#%%############################################################################
def save_obj(obj, name ):
    f = gzip.open(name + '.pklz','wb')
    pickle.dump(obj, f)
    f.close()
    #f = open(name + ".json", 'w')
    #jsonStr = json.dumps(obj)
    #f.write(jsonStr)
    #f.close()

def load_obj(name ):
    f = gzip.open(name + '.pklz','rb')
    O = pickle.load(f)
    f.close()
    return O
    #f = open(name +".json", 'r')
    #jsonStr = f.read()
    #data = json.loads(jsonStr)
    #return data

def exists(filename):
    #my_file = Path(filename)
    #if my_file.is_file():
    if os.path.isfile(filename):
        return True
    return False
        
        
def listFilesWithExtension(wdir,ext):
    L = listdir(wdir)
    l=len(ext)
    Lext=[f[:-l] for f in L if f[-l:]==ext]
    Lext.sort()
    return Lext