"""
@author: Shuo Zhou, The University of Sheffield, szhou@sheffield.ac.uk

Ref: Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization: 
A geometric framework for learning from labeled and unlabeled examples. 
Journal of machine learning research, 7(Nov), 2399-2434.
"""

import sys
import warnings
import numpy as np
from numpy.linalg import multi_dot, inv
import scipy.sparse as sparse
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp
from ..utils.multiclass import score2pred


def lapnorm(X, n_neighbour=3, metric='cosine', mode='distance',
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


def solve_semi_dual(K, y, Q_, C, solver='osqp'):
    if len(y.shape) == 1:
        coef_, support_ = semi_binary_dual(K, y, Q_, C, solver)
    else:
        coef_list = []
        support_ = []
        for i in range(y.shape[1]):
            coef, support = semi_binary_dual(K, y[:, i], Q_, C, solver)
            coef_list.append(coef.reshape(-1, 1))
            support_.append(support)

        coef_ = np.concatenate(coef_list, axis=1)

    return coef_, support_


def semi_binary_dual(K, y_, Q_, C, solver='osqp'):
    """
    Construct & solve quraprog problem
    :param K:
    :param y_:
    :param Q_:
    :param C:
    :param solver:
    :return:
    """
    nl = y_.shape[0]
    n = K.shape[0]
    J = np.zeros((nl, n))
    J[:nl, :nl] = np.eye(nl)
    Q_inv = inv(Q_)
    Y = np.diag(y_.reshape(-1))
    Q = multi_dot([Y, J, K, Q_inv, J.T, Y])
    Q = Q.astype('float32')
    alpha = _quadprog(Q, y_, C, solver)
    coef_ = multi_dot([Q_inv, J.T, Y, alpha])
    support_ = np.where((alpha > 0) & (alpha < C))
    return coef_, support_


def _quadprog(Q, y, C, solver='osqp'):
    """
    solve quadratic programming problem
    :param y: Label, array-like, shape (nl_samples, )
    :param Q: Quad matrix, array-like, shape (n_samples, n_samples)
    :return: coefficients alpha
    """
    # dual
    nl = y.shape[0]
    q = -1 * np.ones((nl, 1))

    if solver == 'cvxopt':
        G = np.zeros((2 * nl, nl))
        G[:nl, :] = -1 * np.eye(nl)
        G[nl:, :] = np.eye(nl)
        h = np.zeros((2 * nl, 1))
        h[nl:, :] = C / nl

        # convert numpy matrix to cvxopt matrix
        P = matrix(Q)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(y.reshape(1, -1).astype('float64'))
        b = matrix(np.zeros(1).astype('float64'))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        alpha = np.array(sol['x']).reshape(nl)

    elif solver == 'osqp':
        warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
        P = sparse.csc_matrix((nl, nl))
        P[:nl, :nl] = Q[:nl, :nl]
        G = sparse.vstack([sparse.eye(nl), y.reshape(1, -1)]).tocsc()
        l = np.zeros((nl + 1, 1))
        u = np.zeros(l.shape)
        u[:nl, 0] = C

        prob = osqp.OSQP()
        prob.setup(P, q, G, l, u, verbose=False)
        res = prob.solve()
        alpha = res.x

    else:
        print('Invalid QP solver')
        sys.exit()

    return alpha


def solve_semi_ls(Q, y):
    n = Q.shape[0]
    nl = y.shape[0]
    Q_inv = inv(Q)
    if len(y.shape) == 1:
        y_ = np.zeros(n)
        y_[:nl] = y[:]
    else:
        y_ = np.zeros((n, y.shape[1]))
        y_[:nl, :] = y[:, :]
    return np.dot(Q_inv, y_)


class LapSVM(BaseEstimator, TransformerMixin):
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
        nl = y.shape[0]
        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        I = np.eye(n)
        if self.gamma_ == 0:
            Q_ = I
        else:
            L = lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode)
            Q_ = I + self.gamma_ * np.dot(L, K)

        y_ = self._lb.fit_transform(y)
        self.coef_, self.support_ = solve_semi_dual(K, y_, Q_, self.C, self.solver)
        if self._lb.y_type_ == 'binary':
            self.support_vectors_ = X[:nl, :][self.support_]
            self.n_support_ = self.support_vectors_.shape[0]
        else:
            self.support_vectors_ = []
            self.n_support_ = []
            for i in range(y_.shape[1]):
                self.support_vectors_.append(X[:nl, :][self.support_[i]][-1])
                self.n_support_.append(self.support_vectors_[-1].shape[0])

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
        check_is_fitted(self, 'X')
        check_is_fitted(self, 'y')
        # X_fit = self.X
        K = pairwise_kernels(X, self.X, metric=self.kernel, **self.kwargs)

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


class LapRLS(BaseEstimator, TransformerMixin):
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
        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            L = lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode,
                        metric=self.manifold_metric)
            Q_ = np.dot((J + self.gamma_ * L), K) + self.sigma_ * I
        else:
            Q_ = np.dot(J, K) + self.sigma_ * I

        y_ = self._lb.fit_transform(y)
        self.coef_ = solve_semi_ls(Q_, y_)

        self.X = X
        self.y = y

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
        K = pairwise_kernels(X, self.X, metric=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)

    def fit_predict(self, X, y):
        """
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            y: Label, array-like, shape (nl_samples, ), where nl_samples <= n_samples
        """
        self.fit(X, y)

        return self.predict(X)
