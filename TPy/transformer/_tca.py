# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from scipy.linalg import eig
from numpy.linalg import multi_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
# from sklearn.utils.validation import check_is_fitted
from ..utils import lap_norm, mmd_coef, base_init


class TCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel='linear', lambda_=1.0,
                 mu=1.0, gamma_=0.5, k=3, **kwargs):
        """Transfer Component Analysis: TCA
        
        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= (N, d))
        kernel: str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        lambda_ : float
            regulization param
        mu : float
            KNN graph param
        k : int
            number of nearest neighbour for KNN graph
        gamma : float
            label dependence param

        References
        ----------
        S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
        Transfer Component Analysis," in IEEE Transactions on Neural Networks,
        vol. 22, no. 2, pp. 199-210, Feb. 2011.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_ 
        self.mu = mu
        self.gamma_ = gamma_
        self.k = k
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)

    def fit(self, Xs, Xt, ys=None, yt=None):
        """[summary]
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.

        Parameters
        ----------
        Xs : array-like
            Source domain data, shape (ns_samples, n_features)
        Xt : array-like
            Target domain data, shape (nt_samples, n_features)
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None
        """
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        X = np.vstack((Xs, Xt))
        L = mmd_coef(ns, nt, kind='marginal', mu=0)
        L[np.isnan(L)] = 0

        K, I, H, n = base_init(X, kernel=self.kernel, **self.kwargs)
        
        if ys is not None:
            # semisupervised TCA (SSTCA)
            Ys_ = self._lb.fit_transform(ys)
            n_class = Ys_.shape[1]
            y = np.zeros((n, n_class))
            y[:Ys_.shape[0], :] = Ys_[:]
            if yt is not None:
                Yt_ = self._lb.transform(yt)
                y[Ys_.shape[0]:Yt_.shape[0], :] = Yt_[:]                        
            Kyy = self.gamma_ * np.dot(y, y.T) + (1-self.gamma_) * I
            Lap_ = lap_norm(X, n_neighbour=self.k, mode='connectivity')
            obj = multi_dot([K, (L + self.mu * Lap_), K.T]) + self.lambda_ * I
            st = multi_dot([K, H, Kyy, H, K.T]) + multi_dot([K, H, K.T])
        # obj = np.trace(np.dot(K,L))
        else: 
            obj = multi_dot([K, L, K.T]) + self.lambda_ * I
            st = multi_dot([K, H, K.T])
        eig_val, eig_vec = eig(obj, st)
        idx_sorted = eig_val.argsort()

        self.U = eig_vec[:, idx_sorted]
        self.U = np.asarray(self.U, dtype=np.float)

        self.Xs = Xs
        self.Xt = Xt
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
        # check_is_fitted(self, 'Xs')
        # check_is_fitted(self, 'Xt')
        X_fit = np.vstack((self.Xs, self.Xt))
        K = pairwise_kernels(X, X_fit, metric=self.kernel, filter_params=True, **self.kwargs)
        U_ = self.U[:, :self.n_components]
        return np.dot(K, U_)

    def fit_transform(self, Xs, Xt, ys=None, yt=None):
        """
        Parameters
        ----------
        Xs : array-like
            Source domain data, shape (ns_samples, n_features)
        Xt : array-like
            Target domain data, shape (nt_samples, n_features)
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None

        Returns
        -------
        array-like
            transformed Xs_transformed, Xt_transformed
        """
        self.fit(Xs, Xt, ys, yt)
        Xs_transformed = self.transform(Xs)
        Xt_transformed = self.transform(Xt)
        return Xs_transformed, Xt_transformed
