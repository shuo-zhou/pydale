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

import warnings
import osqp
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import multi_dot
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator


class ASVM(BaseEstimator):
    def __init__(self, C=1.0, clfs=None, t=None, kernel='linear', solver='osqp', **kwargs):
        self.C = C
        self.src_clf = None
        self.kwargs = kwargs
        self.kernel = kernel
        self.clfs = clfs
        if t is None and len(clfs) >= 0:
            self.t = np.zeros(len(clfs))
            self.t[:] = 1 / len(clfs)
        else:
            self.t = t
        self.solver = solver
        self.n_ = None
        self.alpha = None

    def fit(self, X, y):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        :param X: Training data
        :param y: Traning data labels
        :return: self
        """
        self.n_ = X.shape[0]
        
        # create matrix P
        P = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
            
        # create vector q
        q = np.zeros(self.n_)
        for i in range(len(self.clfs)):
            clf = self.clfs[i]
            q += self.t[i] * clf.decision_function(X)

        # create the Matrix of SVM contraints
        G = sparse.eye(self.n_)

        # create vector of h
        h = np.zeros(self.n_)
        h[:self.n_, :] = self.C

        A = matrix(y.reshape(1, -1).astype('float32'))
        b = matrix(np.zeros(1).astype('float32'))

        self.alpha = self.sol_qp(P, q, G, h, A, b)
        self.X = X
        self.y = y

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
    
    def predict(self, X):
        pred = np.sign(self.decision_function(X))
        return pred
        
    def decision_function(self, X):
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')
        dec_src = np.zeros(X.shape[0])
        for i in range(len(self.clfs)):
            clf = self.clfs[i]
            dec_src += self.t[i] * clf.decision_function(X)
        Kx = pairwise_kernels(X, self.X, self.kernel, **self.kwargs)
        dec_delta = multi_dot([Kx, np.multiply(self.alpha, self.y)])
        return dec_src + dec_delta
