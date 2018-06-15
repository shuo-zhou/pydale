# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 22:54:54 2018

@author: sz144
"""

import sys
import math
import numpy as np
from sklearn.datasets import make_blobs
#from sklearn.datasets import make_classification
#import matplotlib.pyplot as plt

def get_rot_matrix(n_features):
    rot_mat = np.zeros((n_features, n_features))
    rot_dirt = np.array([[math.cos(30), - math.sin(30)],[math.sin(30), math.cos(30)]])
    rot_mat[:2, :2] = rot_dirt
    rot_mat[2:, 2:] = np.eye(n_features -2)
#    np.random.seed(seed = 144)
#    rot_rand = np.random.randint(2, size=n_features)
#    rot_rand[np.where(rot_rand==0)] = -1
#    rot_rand = rot_rand.reshape((n_features, 1)) 
#    rot_mat = np.dot(rot_rand, rot_rand.T)
#    rot_mat = rot_mat - np.eye(n_features)
    return rot_mat


def get_toydata(n_samples=100, mismatch='joint', n_features = 3, n_classes = 2):
    if mismatch not in ['joint','marginal']:
        sys.exit('Invalid mismatch type!')
    np.random.seed(seed = 144)
    centres = np.random.normal(size = (n_classes, n_features))
    
    
    Xt, yt = make_blobs(n_samples = n_samples, n_features = n_features, 
                        centers = centres + 5, random_state = 144)

    rot_mat = get_rot_matrix(n_features)
    Xt = np.dot(Xt, rot_mat)
    Xs, ys = make_blobs(n_samples = n_samples, n_features = n_features, 
                        centers = centres, random_state = 14)
    
    return Xs, ys, Xt, yt
 