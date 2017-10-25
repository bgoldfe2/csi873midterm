# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:12:57 2017

@author: Bruce
"""

import numpy as np

a = [1,2,3,4]
npa = np.asarray(a)
b = 3
d = [2,2,3,4]
npd = np.asarray(d)
c = np.multiply(a,b)
print (c)

e = np.multiply(a,d)
print (e)
