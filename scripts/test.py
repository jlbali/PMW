#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:28:28 2018

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


x=np.arange(np.pi/4,7*np.pi/4,.3)#0.01)
X=np.sin(x)
y=np.arange(-2,2,0.2)
Y=sp.stats.norm.pdf(y)+.7#Y=np.cos(y)
M=(X[:, np.newaxis].dot(Y[ np.newaxis,:]))

plt.imshow(M)
plt.hist(M.ravel(),bins=100)
