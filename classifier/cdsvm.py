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
import warnings
import osqp
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import multi_dot
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator


class CDSVM(BaseEstimator):
    def __init__(self, X_src, y_src, C=0.1, beta=0.5, kernel='linear', solver='osqp', **kwargs):
        self.C = C
        self.beta = beta
        self.X_src = X_src
        self.y_src = y_src
        self.solver = solver
        self.n_ = None
        self.alpha = None
        self.kwargs = kwargs
        self.kernel = kernel
        
    def fit(self, X, y):
        n_support = len(self.y_src)
        X_all = np.concatenate((self.X_src, X))
        y_all = np.concatenate((self.y_src, y))
        n_ = X_all.shape[0]

        # create matrix P
        P = pairwise_kernels(X_all, metric=self.kernel, **self.kwargs)

        # create vector q
        q = np.zeros((n_, 1))

        # create the Matrix of SVM contraints
        G = sparse.eye(n_)

        # create vector of h
        h = np.zeros(n_)
        for i in range(n_support):
            h[i] = self.sigma(self.X_src[i, :], X) * self.C
        h[n_support:, :] = self.C

        A = matrix(y.reshape(1, -1).astype('float32'))
        b = matrix(np.zeros(1).astype('float32'))

        self.alpha = self.sol_qp(P, q, G, h, A, b)
        self.X = X_all
        self.y = y_all

    def sigma(self, support_vector, X):
        n_samples = X.shape[0]
        sigma = 0
        for i in range(n_samples):
            sigma += np.exp(-self.beta * np.linalg.norm(support_vector-X[i, :]))
        sigma = sigma / n_samples
        return sigma
        
    def predict(self, X):
        pred = np.sign(self.decision_function(X))
        return pred
        
    def decision_function(self, X):
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')
        Kx = pairwise_kernels(X, self.X, self.kernel, **self.kwargs)
        dec = multi_dot([Kx, np.multiply(self.alpha, self.y)])

        return dec

    def sol_qp(self, P, q, G, h, A, b):
        if self.solver == 'osqp':
            warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
            P = sparse.csc_matrix(P)
            G = sparse.vstack([sparse.eye(G), A]).tocsc()
            l = np.zeros((h.shape[0] + 1, 1))
            u = np.zeros(l.shape)
            u[:h.shape[0], 0] = h[:]

            prob = osqp.OSQP()
            prob.setup(P, q, G, l, u, verbose=False)
            res = prob.solve()
            alpha = res.x

        elif self.solver == 'cvxopt':
            P = matrix(P)
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)
            A = matrix(A)
            b = matrix(b)

            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)
            alpha = np.array(sol['x']).reshape(P.shape[0])

        return alpha

