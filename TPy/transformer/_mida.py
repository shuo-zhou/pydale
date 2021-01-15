# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================

import numpy as np
from scipy.linalg import eig
from numpy.linalg import multi_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
from ..utils import base_init
# from sklearn.neighbors import kneighbors_graph


class MIDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, penalty=None, kernel='linear', lambda_=1.0, 
                 mu=1.0, eta=1.0, aug=True, **kwargs):
        """Maximum independence domain adaptation
        
        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= d)
        kernel : str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        penalty : str
            None | 'l2' (default is None)
        lambda_ : float
            regulization param (if penalty==l2)
        mu: total captured variance param
        eta: label dependence param
            
        References
        ----------
        Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace 
        using domain features and independence maximization. IEEE transactions on 
        cybernetics, 48(1), pp.288-299.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.lambda_ = lambda_
        self.penalty = penalty
        self.mu = mu
        self.eta = eta
        self.aug = aug
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)
        self.kwargs = kwargs

    def fit(self, X, y=None, co_variates=None):
        """
        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Labels, shape (nl_samples,)
        co_variates : array-like
            Domain co-variates, shape (n_samples, n_co-variates)

        Note:
            Unsupervised MIDA is performed if ys and yt are not given.
            Semi-supervised MIDA is performed is ys and yt are given.
        """
        
        K, I, H, n = base_init(X, kernel=self.kernel, **self.kwargs)
        if type(co_variates) == np.ndarray:
            Kd = np.dot(co_variates, co_variates.T)
        else:
            Kd = np.zeros((n, n))
        if y is not None:
            y_ = self._lb.fit_transform(y)
            Ky = np.dot(y_, y_.T)
            obj = multi_dot([K, H, Kd, H, K.T])
            st = multi_dot([K, H, (self.mu * I + self.eta * Ky), H, K.T])
        # obj = np.trace(np.dot(K,L))
        else: 
            obj = multi_dot([K, H, Kd, H, K.T])
            st = multi_dot([K, H, K.T])
            
        if self.penalty == 'l2':
            obj -= self.lambda_ * I

        eig_values, eig_vectors = eig(obj, st)
        idx_sorted = eig_values.argsort()

        self.U = eig_vectors[:, idx_sorted]
        self.U = np.asarray(self.U, dtype=np.float)
#        self.components_ = np.dot(X.T, U)
#        self.components_ = self.components_.T

        self.X = X
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)

        Returns
        -------
        array-like
            transformed data
        """
        check_is_fitted(self, 'X')
        X_fit = self.X
        K = pairwise_kernels(X, X_fit, metric=self.kernel, filter_params=True, **self.kwargs)
        U_ = self.U[:, :self.n_components]
        X_transformed = np.dot(K, U_)
        return X_transformed

    def fit_transform(self, X, y=None, co_variates=None):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)
        y : array-like
            shape (n_samples,)
        co_variates : array-like
            shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            transformed X_transformed
        """
        self.fit(X, y, co_variates)
        X_transformed = self.transform(X)
        if self.aug:
            X_transformed = np.concatenate((X_transformed, co_variates), axis=1)
        return X_transformed
