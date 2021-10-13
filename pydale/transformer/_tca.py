# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from scipy.linalg import eig
from numpy.linalg import multi_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer, LabelBinarizer
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
            regulisation param
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
        self._centerer = KernelCenterer()

    def fit(self, xs, ys=None, xt=None, yt=None):
        """[summary]
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.

        Parameters
        ----------
        xs : array-like
            Source domain data, shape (ns_samples, n_features)
        xt : array-like
            Target domain data, shape (nt_samples, n_features)
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None
        """
        if type(xt) == np.ndarray:
            x = np.vstack((xs, xt))
            ns = xs.shape[0]
            nt = xt.shape[0]
            L = mmd_coef(ns, nt, kind='marginal', mu=0)
            L[np.isnan(L)] = 0
        else:
            x = xs.copy()
            L = np.zeros((x.shape[0], x.shape[0]))

        krnl_x, unit_mat, ctr_mat, n = base_init(x, kernel=self.kernel, **self.kwargs)
        krnl_x = self._centerer.fit_transform(krnl_x)

        obj = self.lambda_ * unit_mat
        st = multi_dot([krnl_x, ctr_mat, krnl_x.T])
        if ys is not None:
            # semisupervised TCA (SSTCA)
            ys_mat = self._lb.fit_transform(ys)
            n_class = ys_mat.shape[1]
            y = np.zeros((n, n_class))
            y[:ys_mat.shape[0], :] = ys_mat[:]
            if yt is not None:
                yt_mat = self._lb.transform(yt)
                y[ys_mat.shape[0]:yt_mat.shape[0], :] = yt_mat[:]
            ker_y = self.gamma_ * np.dot(y, y.T) + (1 - self.gamma_) * unit_mat
            lap_mat = lap_norm(x, n_neighbour=self.k, mode='connectivity')
            obj += multi_dot([krnl_x, (L + self.mu * lap_mat), krnl_x.T])
            st += multi_dot([krnl_x, ctr_mat, ker_y, ctr_mat, krnl_x.T])
        # obj = np.trace(np.dot(krnl_x,L))
        else: 
            obj += multi_dot([krnl_x, L, krnl_x.T])

        eig_values, eig_vectors = eig(obj, st)
        idx_sorted = eig_values.argsort()

        self.U = np.asarray(eig_vectors[:, idx_sorted], dtype=np.float)
        self.xs = xs
        self.xt = xt

        return self

    def transform(self, x):
        """
        Parameters
        ----------
        x : array-like,
            shape (n_samples, n_features)

        Returns
        -------
        array-like
            transformed data
        """
        # check_is_fitted(self, 'Xs')
        # check_is_fitted(self, 'Xt')
        x_fit = np.vstack((self.xs, self.xt))
        ker_x = pairwise_kernels(x, x_fit, metric=self.kernel,
                                 filter_params=True, **self.kwargs)

        return np.dot(ker_x, self.U[:, :self.n_components])

    def fit_transform(self, xs, ys=None, xt=None, yt=None):
        """
        Parameters
        ----------
        xs : array-like
            Source domain data, shape (ns_samples, n_features).
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None.
        xt : array-like
            Target domain data, shape (nt_samples, n_features), by default None.
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None.

        Returns
        -------
        array-like
            transformed Xs_transformed, Xt_transformed
        """
        self.fit(xs, ys, xt, yt)

        return self.transform(xs), self.transform(xt)
