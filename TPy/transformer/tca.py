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
from ..utils.mmd import mmd_coef
from ..estimator.manifold_learn import lapnorm
# =============================================================================
# Transfer Component Analysis: TCA
# Ref: S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via 
# Transfer Component Analysis," in IEEE Transactions on Neural Networks, 
# vol. 22, no. 2, pp. 199-210, Feb. 2011.
# =============================================================================


class TCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel='linear', lambda_=1,
                 mu=1, gamma_=0.5, k=3, **kwargs):
        """
        Parameters
            n_components: n_componentss after tca (n_components <= (N, d))
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            lambda_: regulization param
            mu: KNN graph param
            k: number of nearest neighbour for KNN graph
            gamma: label dependence param
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
        """
        Parameters:
            Xs: Source domain data, array-like, shape (ns_samples, n_feautres)
            Xt: Target domain data, array-like, shape (nt_samples, n_feautres)
            ys: Source domain labels, array-like, shape (ns_samples,)
            yt: Target domain labels, array-like, shape (nt_samples,)
        Note:
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.
        """
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = ns + nt
        X = np.vstack((Xs, Xt))
        L = mmd_coef(ns, nt, kind='marginal', mu=0)
        L[np.isnan(L)] = 0
        K = pairwise_kernels(X, metric=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0
        
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))

        if ys is not None and yt is not None:
            Ys_ = self._lb.fit_transform(ys)
            Yt_ = self._lb.transform(yt)
            y = np.concatenate((Ys_, Yt_), axis=0)
            Kyy = self.gamma_ * np.dot(y, y.T) + (1-self.gamma_) * I
            Lap_ = lapnorm(X, n_neighbour=self.k, mode='connectivity')
            obj = multi_dot([K, (L + self.mu * Lap_), K.T]) + self.lambda_ * I
            st = multi_dot([K, H, Kyy, H, K.T])
        # obj = np.trace(np.dot(K,L))
        else: 
            obj = multi_dot([K, L, K.T]) + self.lambda_ * I
            st = multi_dot([K, H, K.T])
        eig_vals, eig_vecs = eig(obj, st)
        
#        ev_abs = np.array(list(map(lambda item: np.abs(item), eig_vals)))
#        idx_sorted = np.argsort(ev_abs)
        idx_sorted = eig_vals.argsort()

        self.eig_vals = eig_vals[idx_sorted]
        self.U = eig_vecs[:, idx_sorted]
        self.U = np.asarray(self.U, dtype = np.float)
#        self.components_ = np.dot(X.T, U)
#        self.components_ = self.components_.T

        self.Xs = Xs
        self.Xt = Xt
        return self

    def transform(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            tranformed data
        """
        # check_is_fitted(self, 'Xs')
        # check_is_fitted(self, 'Xt')
        X_fit = np.vstack((self.Xs, self.Xt))
        K = pairwise_kernels(X, X_fit, metric=self.kernel, **self.kwargs)
        U_ = self.U[:, :self.n_components]
        X_transformed = np.dot(K, U_)
        return X_transformed

    def fit_transform(self, Xs, Xt, ys=None, yt=None):
        """
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
        Return:
            tranformed Xs_transformed, Xt_transformed
        """
        self.fit(Xs, Xt, ys, yt)
        Xs_transformed = self.transform(Xs)
        Xt_transformed = self.transform(Xt)
        return Xs_transformed, Xt_transformed
