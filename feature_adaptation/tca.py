# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================

import numpy as np
import scipy.linalg
import sys
from sklearn.metrics.pairwise import kernel_metrics

# =============================================================================
# Transfer Component Analysis: TCA
# Ref: S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
# Transfer Component Analysis," in IEEE Transactions on Neural Networks,
# vol. 22, no. 2, pp. 199-210, Feb. 2011.
# =============================================================================

class TCA:
    def __init__(self, n_components, kernel_type='linear', lambda_=1, **kwargs):
        '''
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel_type: 'rbf' | 'linear' | 'poly' (default is 'linear')
            kernelparam: kernel param
            lambda_: regulization param
        '''
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel_type = kernel_type
        self.lambda_ = lambda_

    def get_L(self, ns, nt):
        '''
        Get kernel weight matrix
        Parameters:
            ns: source domain sample size
            nt: target domain sample size
        Return:
            Kernel weight matrix L
        '''
        a = 1.0 / (ns * np.ones((ns, 1)))
        b = -1.0 / (nt * np.ones((nt, 1)))
        e = np.vstack((a, b))
        L = np.dot(e, e.T)
        return L

    def get_kernel(self, X, Y=None):
        '''
        Generate kernel matrix
        Parameters:
            X: X matrix (n1,d)
            Y: Y matrix (n2,d)
        Return:
            Kernel matrix K
        '''
        kernel_all = ['linear', 'rbf', 'poly']
        if self.kernel_type not in kernel_all:
            sys.exit('Invalid kernel type!')
        kernel_function = kernel_metrics()[self.kernel_type]
        return kernel_function(X, Y=Y, **self.kwargs)


    def fit(self, Xs, Xt):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = ns + nt
        X = np.vstack((Xs, Xt))
        L = self.get_L(ns, nt)
        L[np.isnan(L)] = 0
        K = self.get_kernel(X)
        K[np.isnan(K)] = 0
        #obj = np.trace(np.dot(K,L))

        H = np.eye(n) - 1. / n * np.ones((n, n))

        obj = np.dot(np.dot(K, L), K.T) + self.lambda_ * np.eye(n)
        st = np.dot(np.dot(K, H), K.T)
        eig_values, eig_vecs = scipy.linalg.eig(obj, st)

        ev_abs = np.array(list(map(lambda item: np.abs(item), eig_values)))
        idx_sorted = np.argsort(ev_abs)[:self.n_components]


        U = np.zeros((eig_vecs.shape[0], self.n_components))

        U[:,:] = eig_vecs[:, idx_sorted]
        self.U = np.asarray(U, dtype = np.float)
#        self.components_ = np.dot(X.T, U)
#        self.components_ = self.components_.T
        self.K = K
        self.Xs = Xs
        self.Xt = Xt
        return self

    def transform(self, X):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            tranformed data
        '''
        check_is_fitted(self, 'Xs')
        check_is_fitted(self, 'Xt')
        X_fit = np.vstack(self.Xs, self.Xt)
        K = self.get_kernel(X, X_fit)
        X_transformed = np.dot(K, self.U)
        return X_transformed


    def fit_transform(self, Xs, Xt):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
        Return:
            tranformed Xs_transformed, Xt_transformed
        '''
        self.fit(Xs, Xt)
        K_ = np.dot(self.K, self.U)
        Xs_transformed = K_[:self.ns, :]
        Xt_transformed = K_[self.ns:, :]
        return Xs_transformed, Xt_transformed
