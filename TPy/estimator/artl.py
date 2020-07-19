# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk, The University of Sheffield
# =============================================================================
import sys
import warnings
import numpy as np
import scipy.sparse as sparse
from numpy.linalg import multi_dot, inv, solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp
from .manifold_learn import lapnorm, semi_binary_dual, semi_binary_ls
from ..utils.mmd import mmd_coef
from ..utils.multiclass import score2pred

# =============================================================================
# Adaptation Regularisation Transfer Learning: ARTL
# Ref: Long, M., Wang, J., Ding, G., Pan, S.J. and Philip, S.Y., 2013. 
# Adaptation regularization: A general framework for transfer learning. 
# IEEE Transactions on Knowledge and Data Engineering, 26(5), pp.1076-1089.
# =============================================================================


def cat_xy(Xs, ys, Xt, yt=None):
    if yt is not None:
        X = np.concatenate([Xs, Xt], axis=0)
        y = np.concatenate([ys, yt])
    else:
        X = np.concatenate([Xs, Xt], axis=0)
        y = ys.copy()

    return X, y


class ARSVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, gamma_=0, k_neighbour=5,
                 solver='osqp', manifold_metric='cosine', knn_mode='distance', **kwargs):
        """
        Parameters
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            lambda_: MMD regulization param
            gamma_: manifold regulization param
            solver: osqp (default), cvxopt
            kwargs: kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = C
        self.gamma_ = gamma_
        self.solver = solver
        self.k_neighbour = k_neighbour
        # self.alpha = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)
        # self.scaler = StandardScaler()

    def fit(self, Xs, ys, Xt, yt=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xt: Target data, array-like, shape (nt_samples, n_feautres),
                the first ntl samples are labelled if yt is not None
            yt: Target label, array-like, shape (ntl_samples, )
        """
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = ns + nt
        X, y = cat_xy(Xs, ys, Xt, yt)
        nl = y.shape[0]  # number of labelled data

        M = mmd_coef(ns, nt, ys, yt, kind='joint')

        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        y_ = self._lb.fit_transform(y)

        # dual
        I = np.eye(n)

        if self.gamma_ != 0:
            L = lapnorm(X, n_neighbour=self.n_neighbour, mode=self.knn_mode)
            Q_ = I + multi_dot([(self.lambda_ * M + self.gamma_ * L), K])
        else:
            Q_ = I + multi_dot([(self.lambda_ * M), K])
        Q_inv = inv(Q_)

        if self._lb.classes_.shape[0] == 2:
            self.coef_, self.support_ = semi_binary_dual(K, y_, Q_inv,
                                                         self.C, self.solver)
            self.support_vectors_ = X[:nl, :][self.support_]
            self.n_support_ = self.support_vectors_.shape[0]

        else:
            coef_list = []
            self.support_ = []
            self.support_vectors_ = []
            self.n_support_ = []
            for i in range(y_.shape[1]):
                coef_, support_ = semi_binary_dual(K, y_[:, i], Q_inv,
                                                   self.C, self.solver)
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
        n_class = self._lb.classes_.shape[0]
        if n_class == 2:
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, Xs, ys, Xt, yt=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xt: Target data, array-like, shape (nt_samples, n_feautres),
                the first ntl samples are labelled if yt is not None
            yt: Target label, array-like, shape (ntl_samples, )
        """
        self.fit(Xs, ys, Xt, yt)
        if yt is not None:
            ntl = yt.shape[0]
            Xtest = Xt[ntl:, :]
        else:
            Xtest = Xt
        return self.predict(Xtest)


class ARRLS(BaseEstimator, TransformerMixin):
    def __init__(self, kernel='linear', lambda_=1, gamma_=0, sigma_=1, k_neighbour=5,
                 manifold_metric='cosine', knn_mode='distance', **kwargs):
        """
        Init function
        Parameters
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            lambda_: MMD regularisation param
            gamma_: manifold regularisation param
            sigma_: l2 regularisation param
            solver: osqp (default), cvxopt
            kwargs: kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.sigma_ = sigma_
        self.k_neighbour = k_neighbour
        # self.coef_ = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, Xs, ys, Xt, yt=None):
        """
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xt: Unlabelled target data,  array-like, shape (ntu_samples, n_feautres)
            Xtl: Labelled target data, array-like, shape (ntl_samples, n_feautres)
            yt: Target label, array-like, shape (ntl_samples, )
        """
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = ns + nt
        X, y = cat_xy(Xs, ys, Xt, yt)
        nl = y.shape[0]  # number of labelled data

        M = mmd_coef(ns, nt, ys, yt, kind='joint')

        I = np.eye(n)
        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            L = lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode,
                        metric=self.manifold_metric)
            Q_ = np.dot((J + self.lambda_ * M + self.gamma_ * L),
                        K) + self.sigma_ * I
        else:
            Q_ = np.dot((J + self.lambda_ * M), K) + self.sigma_ * I
        Q_inv = inv(Q_)

        y_ = self._lb.fit_transform(y)
        if self._lb.classes_.shape[0] == 2:
            self.coef_ = semi_binary_ls(Q_inv, y_)
        else:
            coefs_ = []
            for i in range(y_.shape[1]):
                coefs_.append(semi_binary_ls(Q_inv, y_[:, i]).reshape(-1, 1))
            self.coef_ = np.concatenate(coefs_, axis=1)
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
        n_class = self._lb.classes_.shape[0]
        if n_class == 2:
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

    def fit_predict(self, Xs, ys, Xt, yt=None):
        """
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xt: Unlabelled target data,  array-like, shape (nt_samples, n_feautres)
            yt: Target label, array-like, shape (ntl_samples, )
        """
        self.fit(Xs, ys, Xt, yt)
        if yt is not None:
            ntl = yt.shape[0]
            Xtest = Xt[ntl:, :]
        else:
            Xtest = Xt
        return self.predict(Xtest)
