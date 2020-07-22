"""
@author: Shuo Zhou, The University of Sheffield, szhou@sheffield.ac.uk

Ref: Zhou, S., Li, W., Cox, C.R. and Lu, H., 2020. Side Information Dependence
 as a Regulariser for Analyzing Human Brain Conditions across Cognitive Experiments.
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020).
"""

import sys
import warnings
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import multi_dot, inv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp
from .manifold_learn import lapnorm, semi_binary_dual, solve_semi_ls
from ..utils.multiclass import score2pred


class SIDeRSVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, mu=0, k_neighbour=3,
                 manifold_metric='cosine', knn_mode='distance', solver='osqp', **kwargs):
        """
        Parameters
            C: param for importance of slack variable
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            **kwargs: kernel param
            lambda_: param for side information dependence regularisation
            mu: param for manifold regularisation (default 0, not apply)
            manifold_metric: metric for manifold regularisation
            k: number of nearest numbers for manifold regularisation
            knn_mode: default distance
            solver: quadratic programming solver, cvxopt, osqp (default)
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.C = C
        self.solver = solver
        # self.scaler = StandardScaler()
        # self.coef_ = None
        # self.X = None
        # self.y = None
        # self.support_ = None
        # self.support_vectors_ = None
        # self.n_support_ = None
        self.manifold_metric = manifold_metric
        self.k_neighbour = k_neighbour
        self.knn_mode = knn_mode
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, D):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        """
        # X, D = cat_data(Xl, Dl, Xu, Du)
        # X = self.scaler.fit_transform(X)
        n = X.shape[0]
        nl = y.shape[0]
        Ka = np.dot(D, D.T)
        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        y_ = self._lb.fit_transform(y)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        if self.mu != 0:
            lap_norm = lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode,
                               metric=self.manifold_metric)
            Q_ = I + np.dot(self.lambda_ / np.square(n - 1) * multi_dot([H, Ka, H])
                            + self.mu / np.square(n) * lap_norm, K)
        else:
            Q_ = I + self.lambda_ / np.square(n - 1) * multi_dot([H, Ka, H, K])

        if self._lb.y_type_ == 'binary':
            self.coef_, self.support_ = semi_binary_dual(K, y_, Q_, self.C,
                                                         self.solver)
            self.support_vectors_ = X[:nl, :][self.support_]
            self.n_support_ = self.support_vectors_.shape[0]

        else:
            coef_list = []
            self.support_ = []
            self.support_vectors_ = []
            self.n_support_ = []
            for i in range(y_.shape[1]):
                coef_, support_ = semi_binary_dual(K, y_[:, i], Q_, self.C,
                                                   self.solver)
                coef_list.append(coef_.reshape(-1, 1))
                self.support_.append(support_)
                self.support_vectors_.append(X[:nl, :][support_][-1])
                self.n_support_.append(self.support_vectors_[-1].shape[0])
            self.coef_ = np.concatenate(coef_list, axis=1)

        self.X = X
        self.y = y

        return self

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            decision scores, array-like, shape (n_samples,) for binary
            classification, (n_samples, n_class) for multi-class cases
        """
        K = pairwise_kernels(X, self.X, metric=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples,)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y, D):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        Return:
            predicted labels, array-like, shape (n_samples,)
        """
        self.fit(X, y, D)
        return self.predict(X)


class SIDeRLS(BaseEstimator, TransformerMixin):
    def __init__(self, sigma_=1, lambda_=1, mu=0, kernel='linear', k=3,
                 knn_mode='distance', manifold_metric='cosine',
                 class_weight=None, **kwargs):
        """
        Parameters:
            sigma_: param for model complexity (l2 norm)
            lambda_: param for side information dependence regularisation
            mu: param for manifold regularisation (default 0, not apply)
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            **kwargs: kernel param
            manifold_metric: metric for manifold regularisation
            k: number of nearest numbers for manifold regularisation
            knn_mode: default distance
            class_weight: None | balance (default None)
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.mu = mu
        # self.classes = None
        # self.coef_ = None
        # self.X = None
        # self.y = None
        self.manifold_metric = manifold_metric
        self.k = k
        self.knn_mode = knn_mode
        self.class_weight = class_weight
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, D):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        Return:
            self
        """
        # X, D = cat_data(Xl, Dl, Xu, Du)
        n = X.shape[0]
        nl = y.shape[0]

        Kd = np.dot(D, D.T)
        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))

        if self.mu != 0:
            lap_norm = lapnorm(X, n_neighbour=self.k, mode=self.knn_mode,
                               metric=self.manifold_metric)
            Q_ = self.sigma_ * I + np.dot(J + self.lambda_ / np.square(n - 1)
                                          * multi_dot([H, Kd, H])
                                          + self.mu / np.square(n) * lap_norm, K)
        else:
            Q_ = self.sigma_ * I + np.dot(J + self.lambda_ / np.square(n - 1)
                                          * multi_dot([H, Kd, H]), K)

        y_ = self._lb.fit_transform(y)
        self.coef_ = solve_semi_ls(Q_, y_)

        self.X = X
        self.y = y

        return self

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        
        K = pairwise_kernels(X, self.X, metric=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y, D):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ) where nl_samples <= n_samples
            D: Domain covariate matrix for input data, array-like, shape (n_samples, n_covariates)
        Return:
            predicted labels, array-like, shape (n_samples,)
        """
        self.fit(X, y, D)
        return self.predict(X)

    # def sol_prob(self, X, y, D):
    #     n = X.shape[0]
    #     nl = y.shape[0]
    #     Kd = np.dot(D, D.T)
    #     K = pairwise_kernels(X, kernel=self.kernel, **self.kwargs)
    #     K[np.isnan(K)] = 0
    #
    #     J = np.zeros((n, n))
    #     J[:nl, :nl] = np.eye(nl)
    #
    #     I = np.eye(n)
    #     H = I - 1. / n * np.ones((n, n))
    #
    #     if self.class_weight == 'balance':
    #         n_pos = np.count_nonzero(y == 1)
    #         n_neg = np.count_nonzero(y == -1)
    #         e = np.zeros(n)
    #         e[np.where(y == 1)] = n_neg / nl
    #         e[np.where(y == -1)] = n_pos / nl
    #         E = np.diag(e)
    #     else:
    #         E = J.copy()
    #
    #     Q_ = multi_dot([E, J, K]) + self.sigma_ * I + self.lambda_ * multi_dot([H, Kd, H, K]) / np.square(n - 1)
    #     if self.mu != 0:
    #         lap_norm = lapnorm(X, n_neighbour=self.k, mode=self.knn_mode,
    #                            metric=self.manifold_metric)
    #         Q_ = Q_ + self.mu / np.square(n) * np.dot(lap_norm, K)
    #     Q_inv = inv(Q_)
    #
    #     y_ = np.zeros(n)
    #     y_[:nl] = y[:]
    #     return multi_dot([Q_inv, E, y_])


# def cat_data(Xl, Dl, Xu=None, Du=None):
#     if Xu is not None and Du is not None:
#         X = np.concatenate((Xl, Xu))
#         D = np.concatenate((Dl, Du))
#     else:
#         X = Xl
#         D = Dl
#     return X, D
