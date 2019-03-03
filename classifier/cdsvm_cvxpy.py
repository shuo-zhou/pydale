#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:55:11 2017

@author: Shuo Zhou, University of Sheffield

Linear Cross domain SVM

Ref: Jiang, W., Zavesky, E., Chang, S.-F., and Loui, A. Cross-domain learning methods 
    for high-level visual concept classification. In Image Processing, 2008. ICIP
    2008. 15th IEEE International Conference on (2008), IEEE, pp. 161-164.
"""
import numpy as np
#from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import cvxpy as cp
from scipy import sparse

class CDSVM(BaseEstimator):
    def __init__(self,support_vectors, support_vector_labels,C=0.1, beta=0.5):
        self.C = C
        self.beta = beta
        self.support_vectors = support_vectors
        self.support_vector_labels = support_vector_labels
        
    def fit(self,X,y):
        n_support = len(self.support_vector_labels)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        X = X.reshape((n_samples,n_features))
        y = y.reshape((n_samples,1))
        self.support_vector_labels = self.support_vector_labels.reshape((n_support,1))
       
        paramCount = n_support + n_samples + n_features
        
        #create matrix P
        P = sparse.lil_matrix((paramCount,paramCount))
        P[:n_features, :n_features] = sparse.eye(n_features)
        
        # create vector q
        q = sparse.lil_matrix((paramCount, 1))
        q[n_features: (n_features + n_samples), 0] = self.C * 1
        
        # create the Matrix of SVM contraints
        G = sparse.lil_matrix((n_samples*2,paramCount))
        G[:n_samples,:n_features] = -np.multiply(X, y)
        G[:, n_features: (n_features+n_samples)] = - sparse.vstack((sparse.eye(n_samples), 
                sparse.eye(n_samples)))
        G_ = sparse.lil_matrix((n_support*2,paramCount))
        G_[:n_support, :n_features] = -np.multiply(
                self.support_vectors, self.support_vector_labels)
        G_[:, (n_features+n_samples):] = - sparse.vstack((sparse.eye(n_support), 
                sparse.eye(n_support)))
        G = sparse.vstack([G,G_])
            
        #create vector of h
        h = sparse.lil_matrix(((n_samples+n_support) * 2,1))
        h[:n_samples] = -1
        h[(n_samples*2):(n_samples*2+n_support)] = -1
        
        # construct quadprog problem
        x = cp.Variable((paramCount,1))
        objective = cp.Minimize(cp.quad_form(x, P) + q.transpose() * x)
        constraints = [G * x <= h]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.coef_ = x.value[:n_features].reshape((1, n_features))
        
    def sigma(self,support_vector, X):
        n_samples = X.shape[0]
        sigma = 0
        for i in range(n_samples):
            sigma +=  np.exp(-self.beta * np.linalg.norm(support_vector-X[i,:]))
        sigma = sigma / n_samples
        return sigma
        
    def predict(self,X):
        pred = np.sign(self.decision_function(X))
        
        return pred
        
    def decision_function(self,X):
        decision = np.dot(X,self.coef_.T)

        return decision[:,0]
    
    def score(self,X,y):
        pred = self.predict(X)
        return accuracy_score(pred,y)
    
    def get_params(self,deep=True):
        out = {'support_vectors':self.support_vectors, 'support_vector_labels':
            self.support_vector_labels, 'C': self.C, 'beta': self.beta}
        return out
