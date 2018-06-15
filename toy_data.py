#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:42:53 2018

@author: shuoz
"""

import numpy as np



def get_toydata(n, dim=3):
    np.random.seed(seed = 14)
    
    n_c1 = int(n / 2) # number of samples in class 1
    n_c2 = n - n_c1
    size_c1 = (n_c1, dim)
    size_c2 = (n_c2, dim)
    
#    shape_c1 = np.arange(dim)
#    shape_c2 = np.arange(dim*2, step=2)
    loc_c1 = np.arange(dim) 
    loc_c2 = np.arange(dim) + 2
    
    Xs_c1 = np.random.normal(loc_c1, scale = 0.7, size = size_c1)
    Xs_c2 = np.random.normal(loc_c2, scale = 0.7, size = size_c2)
    Xs = np.vstack((Xs_c1, Xs_c2))
    
    ys = np.zeros(n)
    ys[:n_c1] = 1
    
    loc_c1 = np.arange(dim) + 4
    loc_c2 = np.arange(dim) + 6
    
    Xt_c1 = np.random.normal(loc_c1, scale = 0.7, size = size_c1)
    Xt_c2 = np.random.normal(loc_c2, scale = 0.7, size = size_c2)
    Xt = np.vstack((Xt_c1, Xt_c2))
    yt = np.zeros(n)
    yt[:n_c1] = 1
    
    return Xt, yt, Xs, ys