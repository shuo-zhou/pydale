# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 12:02:32 2017

@author: Shuo Zhou, University of Sheffield

Adaptive Support Vector Machine

ref: Yang, J., Yan, R., & Hauptmann, A. G. (2007, September). 
Cross-domain video concept detection using adaptive svms. 
In Proceedings of the 15th ACM international conference on Multimedia
 (pp. 188-197). ACM.
"""
import numpy as np
from cvxopt import matrix, solvers

## A linear adaptive svm

class ASVM(object):
    def __init__(self,source_coef, C=1.0):
        self.C = C
        self.source_coef = source_coef
        self.n_features = self.source_coef.shape[1]

        
    def fit(self, X, y):
        self.n_samples = X.shape[0]
        paramCount = self.n_samples+self.n_features;
        
        #create matrix P
        P = np.zeros((paramCount,paramCount))
        P[:self.n_features, :self.n_features] = np.eye(self.n_features)
            
        # create vector q
        q = np.zeros((self.n_features + self.n_samples,1))
        q[self.n_features:, :] = self.C * 1
      
        
        # create the Matrix of SVM contraints
        G = np.zeros((2*self.n_samples, paramCount))
        X = X.reshape((self.n_samples,self.n_features))
        y = y.reshape((self.n_samples,1))
        G[:self.n_samples, :self.n_features] = -np.multiply(X, y)
        G[:, self.n_features: (self.n_features+ self.n_samples)] = -np.vstack((
                np.eye(self.n_samples), np.eye(self.n_samples)))
            
        #create vector of h
        h = np.zeros((2*self.n_samples,1))
        h[:self.n_samples, :] = np.multiply((y, np.dot(self.source_coef, X.T))) -1    
                
        # convert numpy matrix to cvxopt matrix
        P = 2*matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h)
        
        self.coef_ = sol['x'][0:self.n_features]
        self.coef_ = np.array(self.coef_).T
    
    def predict(self,X):
        pred = np.sign(self.decision_function(X))
        return pred
        
    def decision_function(self,X):
        decision = np.dot(X,self.source_w.T)+np.dot(X,self.coef_.T)
        return decision[:,0]
