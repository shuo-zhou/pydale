"""
@author: Shuo Zhou, The University of Sheffield, szhou@sheffield.ac.uk

Ref: Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization: 
A geometric framework for learning from labeled and unlabeled examples. 
Journal of machine learning research, 7(Nov), 2399-2434.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError

from ..utils.multiclass import score2pred
from .base import SSLFramework


class LapSVM(SSLFramework):
    def __init__(self, C=1, kernel='linear', gamma_=1, solver='osqp', k_neighbour=3,
                 manifold_metric='cosine', knn_mode='distance', **kwargs):
        """
        Parameters
            C: param for importance of slack variable
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            gamma_: param for manifold regularisation (default 1)
            **kwargs: kernel param
            manifold_metric: metric for manifold regularisation
            k: number of nearest numbers for manifold regularisation
            knn_mode: default distance
            solver: quadratic programming solver, cvxopt, osqp (default)
        """
        self.C = C
        self.gamma_ = gamma_
        self.kernel = kernel
        self.solver = solver
        self.kwargs = kwargs
        self.manifold_metric = manifold_metric
        self.k_neighbour = k_neighbour
        self.knn_mode = knn_mode
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ), where nl_samples <= n_samples
        Return:
            self
        """
        n = X.shape[0]
        # nl = y.shape[0]
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0

        I = np.eye(n)
        if self.gamma_ == 0:
            Q_ = I
        else:
            L = self._lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode)
            Q_ = I + self.gamma_ * np.dot(L, K)

        y_ = self._lb.fit_transform(y)
        self.coef_, self.support_ = self._solve_semi_dual(K, y_, Q_, self.C, self.solver)
        # if self._lb.y_type_ == 'binary':
        #     self.support_vectors_ = X[:nl, :][self.support_]
        #     self.n_support_ = self.support_vectors_.shape[0]
        # else:
        #     self.support_vectors_ = []
        #     self.n_support_ = []
        #     for i in range(y_.shape[1]):
        #         self.support_vectors_.append(X[:nl, :][self.support_[i]][-1])
        #         self.n_support_.append(self.support_vectors_[-1].shape[0])

        self._X = X
        self._y = y

        return self

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            decision scores, array-like, shape (n_samples,) for binary
            classification, (n_samples, n_class) for multi-class cases
        """
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')
        # X_fit = self._X
        K = pairwise_kernels(X, self._X, metric=self.kernel, filter_params=True, **self.kwargs)

        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples, )
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ), where nl_samples <= n_samples
        """
        self.fit(X, y)
        return self.predict(X)


class LapRLS(SSLFramework):
    def __init__(self, kernel='linear', gamma_=1, sigma_=1, k_neighbour=5,
                 manifold_metric='cosine', knn_mode='distance', **kwargs):
        """
        Init function
        Parameters
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            gamma_: manifold regularisation param
            sigma_: l2 regularisation param
            solver: osqp (default), cvxopt
            kwargs: kernel params
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.gamma_ = gamma_
        self.sigma_ = sigma_
        self.k_neighbour = k_neighbour
        # self.coef_ = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ), where nl_samples <= n_samples
        """
        n = X.shape[0]
        nl = y.shape[0]
        I = np.eye(n)
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            L = self._lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode,
                        metric=self.manifold_metric)
            Q_ = np.dot((J + self.gamma_ * L), K) + self.sigma_ * I
        else:
            Q_ = np.dot(J, K) + self.sigma_ * I

        y_ = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q_, y_)

        self._X = X
        self._y = y

        return self

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

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        K = pairwise_kernels(X, self._X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(K, self.coef_)

    def fit_predict(self, X, y):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ), where nl_samples <= n_samples
        """
        self.fit(X, y)

        return self.predict(X)
