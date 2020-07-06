#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:15:26 2019
Ref: Zhou, S., Li, W., Cox, C.R. and Lu, H., 2020. Side Information Dependence
 as a Regulariser for Analyzing Human Brain Conditions across Cognitive Experiments.
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020).
"""

import sys
import warnings
import numpy as np
from scipy.linalg import sqrtm
import scipy.sparse as sparse
from numpy.linalg import multi_dot, inv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp


def get_kernel(X, Y=None, kernel='linear', **kwargs):
    """
    Generate kernel matrix
    Parameters:
        X: X matrix (n1,d)
        Y: Y matrix (n2,d)
        kernel: 'linear'(default) | 'rbf' | 'poly'
    Return:
        Kernel matrix

    """

    return pairwise_kernels(X, Y=Y, metric=kernel,
                            filter_params=True, **kwargs)


def get_lapmat(X, n_neighbour=3, metric='cosine', mode='distance',
               normalise=True):
    """
    Construct Laplacian matrix
    :param X:
    :param n_neighbour:
    :param metric:
    :param mode:
    :param normalise:
    :return:
    """
    n = X.shape[0]
    knn_graph = kneighbors_graph(X, n_neighbour, metric=metric,
                                 mode=mode).toarray()
    W = np.zeros((n, n))
    knn_idx = np.logical_or(knn_graph, knn_graph.T)
    if mode == 'distance':
        graph_kernel = pairwise_distances(X, metric=metric)
        W[knn_idx] = graph_kernel[knn_idx]
    else:
        W[knn_idx] = 1

    D = np.diag(np.sum(W, axis=1))
    if normalise:
        D_ = inv(sqrtm(D))
        lapmat = np.eye(n) - multi_dot([D_, W, D_])
    else:
        lapmat = D - W
    return lapmat


class SIDeRSVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, mu=0, solver='osqp',
                 manifold_metric='cosine', k=3, knn_mode='distance', **kwargs):
        """
        Init function
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
        self.coef_ = None
        self.X = None
        self.y = None
        self.support_ = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.manifold_metric = manifold_metric
        self.k = k
        self.mode = knn_mode
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X_train, y, D_train, X_test=None, D_test=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X_train: Training data, array-like, shape (n_train_samples, n_feautres)
            y: Label, array-like, shape (n_train_samples, )
            D_train: Domain covariate matrix for training data, array-like, shape (n_train_samples, n_covariates)
            X_test: Testing data, array-like, shape (n_test_samples, n_feautres)
            D_test: Domain covariate matrix for testing data, array-like, shape (n_test_samples, n_covariates)
        """

        if X_test is not None and D_test is not None:
            X = np.concatenate((X_train, X_test))
            D = np.concatenate((D_train, D_test))
        else:
            X = X_train.copy()
            D = D_train.copy()
        # X = self.scaler.fit_transform(X)
        n = X.shape[0]
        Ka = np.dot(D, D.T)
        K = get_kernel(X, kernel=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        y_ = self._lb.fit_transform(y)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        Q_ = np.eye(n) + self.lambda_ / np.square(n - 1) * multi_dot([H, Ka, H, K])
        if self.mu != 0:
            lapmat = get_lapmat(X, n_neighbour=self.k, mode=self.mode,
                                metric=self.manifold_metric)
            Q_ = Q_ + self.mu / np.square(n) * np.dot(lapmat, K)
        Q_inv = inv(Q_)

        if self._lb.classes_.shape[0] == 2:
            self.coef_, self.support_ = self._solve_binary(K, y_, Q_inv)
            self.support_vectors_ = X_train[self.support_]
            self.n_support_ = self.support_vectors_.shape[0]

        else:
            coef_list = []
            self.support_ = []
            self.support_vectors_ = []
            self.n_support_ = []
            for i in range(y_.shape[1]):
                y_temp = y_[:, i]
                coef_, support_ = self._solve_binary(K, y_temp, Q_inv)
                coef_list.append(coef_.reshape(-1, 1))
                self.support_.append(support_)
                self.support_vectors_.append(X_train[support_][-1])
                self.n_support_.append(self.support_vectors_[-1].shape[0])
            self.coef_ = np.concatenate(coef_list, axis=1)

        # K_train = get_kernel(X_train, X, kernel=self.kernel, **self.kwargs)
        # self.intercept_ = np.mean(y[self.support_] - y[self.support_] *
        #                           np.dot(K_train[self.support_], self.coef_))/self.n_support_

        # =============================================================================
        #         beta = cvx.Variable(shape = (2 * n, 1))
        #         objective = cvx.Minimize(cvx.quad_form(beta, P) + q.T * beta)
        #         constraints = [G * beta <= h]
        #         prob = cvx.Problem(objective, constraints)
        #         try:
        #             prob.solve()
        #         except SolverError:
        #             prob.solve(solver = 'SCS')
        #
        #         self.coef_ = beta.value[:n]
        # =============================================================================

        #        a = np.dot(W + self.gamma * multi_dot([H, Ka, H]), self.lambda_*I)
        #        b = np.dot(y, W)
        #        beta = solve(a, b)

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
        # check_is_fitted(self, 'X')
        # check_is_fitted(self, 'y')
        # K = get_kernel(self.scaler.transform(X), self.X,
        #                kernel=self.kernel, **self.kwargs)
        K = get_kernel(X, self.X, kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        dec = self.decision_function(X)
        n_class = self._lb.classes_.shape[0]
        if n_class == 2:
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            n_test = X.shape[0]
            y_pred_ = -1 * np.ones((n_test, n_class))
            dec_sort = np.argsort(dec, axis=1)[:, ::-1]
            for i in range(n_test):
                label_idx = dec_sort[i, 0]
                y_pred_[i, label_idx] = 1

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X_train, y, D_train, X_test=None, D_test=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            X_train: Training data, array-like, shape (n_train_samples, n_feautres)
            y: Label, array-like, shape (n_train_samples, )
            D_train: Domain covariate matrix for training data, array-like, shape (n_train_samples, n_covariates)
            X_test: Testing data, array-like, shape (n_test_samples, n_feautres)
            D_test: Domain covariate matrix for testing data, array-like, shape (n_test_samples, n_covariates)
        Return:
            predicted labels, array-like, shape (n_test_samples,)
        """
        self.fit(X_train, y, D_train, X_test, D_test)
        return self.predict(X_test)

    def _solve_binary(self, K, y_, Q_inv):
        n_train = y_.shape[0]
        n = K.shape[0]
        J = np.zeros((n_train, n))
        J[:n_train, :n_train] = np.eye(n_train)
        q = -1 * np.ones((n_train, 1))
        Y = np.diag(y_.reshape(-1))
        Q = multi_dot([Y, J, K, Q_inv, J.T, Y])
        Q = Q.astype('float32')
        alpha = self._quadprog(Q, y_, q)
        coef_ = multi_dot([Q_inv, J.T, Y, alpha])
        support_ = np.where((alpha > 0) & (alpha < self.C))
        return coef_, support_

    def _quadprog(self, Q, y, q):
        """
        solve quadratic programming problem
        :param y: Label, array-like, shape (n_train_samples, )
        :param Q:
        :param q:
        :return: coefficients alpha
        """
        # dual
        n_train = y.shape[0]

        if self.solver == 'cvxopt':
            G = np.zeros((2 * n_train, n_train))
            G[:n_train, :] = -1 * np.eye(n_train)
            G[n_train:, :] = np.eye(n_train)
            h = np.zeros((2 * n_train, 1))
            h[n_train:, :] = self.C / n_train

            # convert numpy matrix to cvxopt matrix
            P = matrix(Q)
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)
            A = matrix(y.reshape(1, -1).astype('float64'))
            b = matrix(np.zeros(1).astype('float64'))

            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            alpha = np.array(sol['x']).reshape(n_train)

        elif self.solver == 'osqp':
            warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
            P = sparse.csc_matrix((n_train, n_train))
            P[:n_train, :n_train] = Q[:n_train, :n_train]
            G = sparse.vstack([sparse.eye(n_train), y.reshape(1, -1)]).tocsc()
            l = np.zeros((n_train + 1, 1))
            u = np.zeros(l.shape)
            u[:n_train, 0] = self.C

            prob = osqp.OSQP()
            prob.setup(P, q, G, l, u, verbose=False)
            res = prob.solve()
            alpha = res.x

        else:
            print('Invalid QP solver')
            sys.exit()

        return alpha


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
        self.classes = None
        self.coef_ = None
        self.X = None
        self.y = None
        self.manifold_metric = manifold_metric
        self.k = k
        self.mode = knn_mode
        self.class_weight = class_weight
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X_train, y, D_train, X_test=None, D_test=None):
        """
        Parameters:
            X_train: Training data, array-like, shape (n_train_samples, n_feautres)
            y: Label, array-like, shape (n_train_samples, )
            D_train: Domain covariate matrix for training data, array-like, shape (n_train_samples, n_covariates)
            X_test: Testing data, array-like, shape (n_test_samples, n_feautres)
            D_test: Domain covariate matrix for testing data, array-like, shape (n_test_samples, n_covariates)
        Return:
            self
        """
        if X_test is not None and D_test is not None:
            X = np.concatenate((X_train, X_test))
            D = np.concatenate((D_train, D_test))
        else:
            X = X_train.copy()
            D = D_train.copy()

        y_ = self._lb.fit_transform(y)

        if self._lb.classes_.shape[0] == 2:
            self.coef_ = self.sol_prob(X, y, D)
        else:
            coefs_ = []
            for i in range(y_.shape[1]):
                y_temp = y_[:, i]
                coefs_.append(self.sol_prob(X, y_temp, D).reshape(-1, 1))
            self.coef_ = np.concatenate(coefs_, axis=1)
        self.X = X
        self.y = y

        return self

    def sol_prob(self, X, y, D):
        n = X.shape[0]
        n_train = y.shape[0]
        Kd = np.dot(D, D.T)
        K = get_kernel(X, kernel=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:n_train, :n_train] = np.eye(n_train)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))

        if self.class_weight == 'balance':
            n_pos = np.count_nonzero(y == 1)
            n_neg = np.count_nonzero(y == -1)
            e = np.zeros(n)
            e[np.where(y == 1)] = n_neg / n_train
            e[np.where(y == -1)] = n_pos / n_train
            E = np.diag(e)
        else:
            E = J.copy()

        Q_ = multi_dot([E, J, K]) + self.sigma_ * I + self.lambda_ * multi_dot([H, Kd, H, K]) / np.square(n - 1)
        Q_inv = inv(Q_)

        y_ = np.zeros(n)
        y_[:n_train] = y[:]
        return multi_dot([Q_inv, E, y_])

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        # check_is_fitted(self, 'X')
        # check_is_fitted(self, 'y')
        # K = get_kernel(self.scaler.transform(X), self.X,
        #                kernel=self.kernel, **self.kwargs)
        K = get_kernel(X, self.X, kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        dec = self.decision_function(X)
        n_class = self._lb.classes_.shape[0]
        if n_class == 2:
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            n_test = X.shape[0]
            y_pred_ = -1 * np.ones((n_test, n_test))
            dec_sort = np.argsort(dec, axis=1)[:, ::-1]
            for i in range(n_test):
                label_idx = dec_sort[i, 0]
                y_pred_[i, label_idx] = 1

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X_train, y, D_train, X_test=None, D_test=None):
        """
        Parameters:
            X_train: Training data, array-like, shape (n_train_samples, n_feautres)
            y: Label, array-like, shape (n_train_samples, )
            D_train: Domain covariate matrix for training data, array-like, shape (n_train_samples, n_covariates)
            X_test: Testing data, array-like, shape (n_test_samples, n_feautres)
            D_test: Domain covariate matrix for testing data, array-like, shape (n_test_samples, n_covariates)
        Return:
            predicted labels, array-like, shape (n_test_samples,)
        """
        self.fit(X_train, y, D_train, X_test, D_test)
        return self.predict(X_test)
