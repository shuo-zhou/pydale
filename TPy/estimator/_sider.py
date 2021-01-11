"""
@author: Shuo Zhou, The University of Sheffield, szhou@sheffield.ac.uk

Ref: Zhou, S., Li, W., Cox, C.R. and Lu, H., 2020. Side Information Dependence
 as a Regulariser for Analyzing Human Brain Conditions across Cognitive Experiments.
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020).
"""

import numpy as np
from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
# import cvxpy as cvx
# from cvxpy.error import SolverError
from ..utils.multiclass import score2pred
from .base import SSLFramework


class SIDeRSVM(SSLFramework):
    def __init__(self, C=1.0, kernel='linear', lambda_=1.0, mu=0.0, k_neighbour=3,
                 manifold_metric='cosine', knn_mode='distance', solver='osqp', **kwargs):
        """Side Information Dependence Regularised Support Vector Machine

        Parameters
        ----------
        C : float, optional
            param for importance of slack variable, by default 1
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        lambda_ : float, optional
            param for side information dependence regularisation, by default 1
        mu : float, optional
            param for manifold regularisation, by default 0
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation, 
            by default 3
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each 
            sample point. The DistanceMetric class gives a list of available 
            metrics. By default 'cosine'.
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of 
            returned matrix: ‘connectivity’ will return the connectivity 
            matrix with ones and zeros, and ‘distance’ will return the 
            distances between neighbors according to the given metric.
        solver : str, optional
            quadratic programming solver, [cvxopt, osqp], by default 'osqp'
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.C = C
        self.solver = solver
        # self.scaler = StandardScaler()
        # self.coef_ = None
        # self._X = None
        # self._y = None
        # self.support_ = None
        # self.support_vectors_ = None
        # self.n_support_ = None
        self.manifold_metric = manifold_metric
        self.k_neighbour = k_neighbour
        self.knn_mode = knn_mode
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, D):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples
        D : array-like, 
            Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns
        -------
        self
            [description]
        """
        # X, D = cat_data(Xl, Dl, Xu, Du)
        # X = self.scaler.fit_transform(X)
        n = X.shape[0]
        # nl = y.shape[0]
        Ka = np.dot(D, D.T)
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0

        y_ = self._lb.fit_transform(y)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        if self.mu != 0:
            lap_norm = self._lapnorm(X, n_neighbour=self.k_neighbour,
                                     mode=self.knn_mode,
                                     metric=self.manifold_metric)
            Q_ = I + np.dot(self.lambda_ / np.square(n - 1) * multi_dot([H, Ka, H])
                            + self.mu / np.square(n) * lap_norm, K)
        else:
            Q_ = I + self.lambda_ / np.square(n - 1) * multi_dot([H, Ka, H, K])

        self.coef_, self.support_ = self._solve_semi_dual(K, y_, Q_, self.C, self.solver)

        # if self._lb.y_type_ == 'binary':
        #     self.coef_, self.support_ = self._semi_binary_dual(K, y_, Q_,
        #                                                        self.C,
        #                                                        self.solver)
        #     self.support_vectors_ = X[:nl, :][self.support_]
        #     self.n_support_ = self.support_vectors_.shape[0]
        #
        # else:
        #     coef_list = []
        #     self.support_ = []
        #     self.support_vectors_ = []
        #     self.n_support_ = []
        #     for i in range(y_.shape[1]):
        #         coef_, support_ = self._semi_binary_dual(K, y_[:, i], Q_,
        #                                                  self.C,
        #                                                  self.solver)
        #         coef_list.append(coef_.reshape(-1, 1))
        #         self.support_.append(support_)
        #         self.support_vectors_.append(X[:nl, :][support_][-1])
        #         self.n_support_.append(self.support_vectors_[-1].shape[0])
        #     self.coef_ = np.concatenate(coef_list, axis=1)

        self._X = X
        self._y = y

        return self

    def decision_function(self, X):
        """[summary]

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
            
        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification, 
            (n_samples, n_class) for multi-class cases
        """
        K = pairwise_kernels(X, self._X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
            
        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y, D):
        """[summary]

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples
        D : array-like, 
            Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(X, y, D)
        return self.predict(X)


class SIDeRLS(SSLFramework):
    def __init__(self, sigma_=1.0, lambda_=1.0, mu=0.0, kernel='linear', 
                 k=3, knn_mode='distance', manifold_metric='cosine', 
                 class_weight=None, **kwargs):
        """Side Information Dependence Regularised Least Square

        Parameters
        ----------
        sigma_ : float, optional
            param for model complexity (l2 norm), by default 1.0
        lambda_ : float, optional
            param for side information dependence regularisation, by default 1.0
        mu : float, optional
            param for manifold regularisation, by default 0.0
        kernel : str, optional
            [description], by default 'linear'
        k : int, optional
            number of nearest numbers for each sample in manifold regularisation, 
            by default 3
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of 
            returned matrix: ‘connectivity’ will return the connectivity 
            matrix with ones and zeros, and ‘distance’ will return the 
            distances between neighbors according to the given metric.
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each 
            sample point. The DistanceMetric class gives a list of available 
            metrics. By default 'cosine'.
        class_weight : [type], optional
            [description], by default None
        **kwargs: 
            kernel param
        """
        
        self.kwargs = kwargs
        self.kernel = kernel
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.mu = mu
        # self.classes = None
        # self.coef_ = None
        # self._X = None
        # self._y = None
        self.manifold_metric = manifold_metric
        self.k = k
        self.knn_mode = knn_mode
        self.class_weight = class_weight
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, D):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples
        D : array-like, 
            Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns
        -------
        self
            [description]
        """
        # X, D = cat_data(Xl, Dl, Xu, Du)
        n = X.shape[0]
        nl = y.shape[0]

        Kd = np.dot(D, D.T)
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))

        if self.mu != 0:
            lap_norm = self._lapnorm(X, n_neighbour=self.k,
                                     mode=self.knn_mode,
                                     metric=self.manifold_metric)
            Q_ = self.sigma_ * I + np.dot(J + self.lambda_ / np.square(n - 1)
                                          * multi_dot([H, Kd, H])
                                          + self.mu / np.square(n) * lap_norm, K)
        else:
            Q_ = self.sigma_ * I + np.dot(J + self.lambda_ / np.square(n - 1)
                                          * multi_dot([H, Kd, H]), K)

        y_ = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q_, y_)

        self._X = X
        self._y = y

        return self

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
            
        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification, 
            (n_samples, n_class) for multi-class cases
        """
        
        K = pairwise_kernels(X, self._X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
            
        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y, D):
        """Fit the model according to the given training data and then perform
            classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_feautres)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples
        D : array-like, 
            Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
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
