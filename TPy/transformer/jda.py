# =============================================================================
# author: Shuo Zhou, The University of Sheffield
# =============================================================================
import sys
import numpy as np
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from ..utils.mmd import mmd_coef
# from sklearn.preprocessing import StandardScaler
# =============================================================================
# Implementation of three transfer learning methods:
#   1. Transfer Component Analysis: TCA
#   2. Joint Distribution Adaptation: JDA
#   3. Balanced Distribution Adaptation: BDA
# Ref:
# [1] S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
# Transfer Component Analysis," in IEEE Transactions on Neural Networks,
# vol. 22, no. 2, pp. 199-210, Feb. 2011.
# [2] Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu,
# Transfer Feature Learning with Joint Distribution Adaptation, IEEE 
# International Conference on Computer Vision (ICCV), 2013.
# [3] Wang, J., Chen, Y., Hao, S., Feng, W. and Shen, Z., 2017, November. Balanced
# distribution adaptation for transfer learning. In Data Mining (ICDM), 2017
# IEEE International Conference on (pp. 1129-1134). IEEE.
# =============================================================================


class JDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel='linear', lambda_=1, mu=1, **kwargs):
        """
        Parameters
            n_components: n_componentss after (n_components <= min(d, n))
            kernel_type: [‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’,
            ‘cosine’] (default is 'linear')
            **kwargs: kernel param
            lambda_: regulisation param
            mu: >= 0, param for conditional mmd, (mu=0 for TCA, mu=1 for JDA, BDA otherwise)
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu

    def fit(self, Xs, Xt, ys=None, yt=None):
        """
        Parameters:
            Xs: Source domain data, array-like, shape (ns_samples, n_feautres)
            Xt: Target domain data, array-like, shape (nt_samples, n_feautres)
            ys: Labels of source domain samples, shape (ns_samples,)
            yt: Labels of source domain samples, shape (nt_samples,)
        """
        X = np.vstack((Xs, Xt))

        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = ns + nt
        if ys is not None and yt is not None:
            L = mmd_coef(ns, nt, ys, yt, kind='joint', mu=self.mu)
        else:
            L = mmd_coef(ns, nt, kind='marginal', mu=0)

        # Construct kernel matrix
        K = pairwise_kernels(X, metric=self.kernel, filter_params=True, **self.kwargs)
        K[np.isnan(K)] = 0
    
        # Construct centering matrix
        H = np.eye(n) - 1.0 / (n * np.ones([n, n]))
        
        # objective for optimization
        obj = np.dot(np.dot(K, L), K.T) + self.lambda_ * np.eye(n)
        # constraint subject to
        st = np.dot(np.dot(K, H), K.T)
        eig_values, eig_vecs = eig(obj, st)
        
        ev_abs = np.array(list(map(lambda item: np.abs(item), eig_values)))
#        idx_sorted = np.argsort(ev_abs)[:self.n_components]
        idx_sorted = np.argsort(ev_abs)
        
        U = np.zeros(eig_vecs.shape)
        U[:, :] = eig_vecs[:, idx_sorted]
        self.U = np.asarray(U, dtype=np.float)
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
        # X = self.scaler.transform(X)
        # check_is_fitted(self, 'Xs')
        # check_is_fitted(self, 'Xt')
        X_fit = np.vstack((self.Xs, self.Xt))
        K = pairwise_kernels(X, X_fit, metric=self.kernel, filter_params=True, **self.kwargs)
        U_ = self.U[:, :self.n_components]
        X_transformed = np.dot(K, U_)
        return X_transformed
    
    def fit_transform(self, Xs, Xt, ys=None, yt=None):
        """
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
            ys: Labels of source domain samples, shape (n_samples,)
            yt: Labels of source domain samples, shape (n_samples,)
        Return:
            tranformed Xs_transformed, Xt_transformed
        """
        self.fit(Xs, Xt, ys, yt)
        Xs_transformed = self.transform(Xs)
        Xt_transformed = self.transform(Xt)
        return Xs_transformed, Xt_transformed
