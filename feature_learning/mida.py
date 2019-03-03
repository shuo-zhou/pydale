# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================

import numpy as np
from scipy.linalg import eig
from numpy.linalg import multi_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import kneighbors_graph
# =============================================================================
# Ref: Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace 
# using domain features and independence maximization. IEEE transactions on 
# cybernetics, 48(1), pp.288-299.
# =============================================================================

def get_kernel(X, Y=None, kernel = 'linear', **kwargs):
    '''
    Generate kernel matrix
    Parameters:
        X: X matrix (n1,d)
        Y: Y matrix (n2,d)
    Return: 
        Kernel matrix
    '''

    return pairwise_kernels(X, Y=Y, metric = kernel, 
                            filter_params = True, **kwargs)

class MIDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, penalty=None, kernel='linear', lambda_=1, 
                 mu = 1, eta = 1, aug = True, **kwargs):
        '''
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            penalty: None | 'l2' (default is None)
            lambda_: regulization param (if penalty==l2)
            mu: total captured variance param
            k: number of nearest neighbour for KNN graph
            eta: label dependence param
        '''
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.penalty = penalty
        self.mu = mu
        self.eta = eta
        self.aug = aug

    def fit(self, X, D, y = None, **kwargs):
        '''
        Parameters:
            X: Input data, array-like, shape (n_samples, n_feautres)
            D: Domain feature, array-like, shape (nt_samples, n_feautres)
            y: Labels, array-like, shape (ns_samples,) or (ns_samples, n_class)
        Note:
            Unsupervised MIDA is performed if ys and yt are not given.
            Semi-supervised MIDA is performed is ys and yt are given.
        '''
        
        n = X.shape[0]
        K = get_kernel(X, kernel = self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0
        
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))
        Kd = np.dot(D, D.T)
        if y is not None:
            y = y.reshape((n,1))
            Ky = np.dot(y, y.T)
            obj = multi_dot([K, H, Kd, H, K.T])
            st = multi_dot([K, H, (self.mu * I + self.eta*Ky), H, K.T])
        #obj = np.trace(np.dot(K,L))            
        else: 
            obj = multi_dot([K, H, Kd, H, K.T])
            st = multi_dot([K, H, K.T])
            
        if self.penalty == 'l2':
            obj -= self.lambda_ * I
        
        eig_vals, eig_vecs = eig(obj, st)
        idx_sorted = eig_vals.argsort()

       
        self.eig_vals = eig_vals[idx_sorted]
        self.U = eig_vecs[:, idx_sorted]
        self.U = np.asarray(self.U, dtype = np.float)
#        self.components_ = np.dot(X.T, U)
#        self.components_ = self.components_.T

        self.X = X
        return self

    def transform(self, X):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            tranformed data
        '''
        check_is_fitted(self, 'X')
        X_fit = self.X
        K = get_kernel(X, X_fit, kernel = self.kernel, **self.kwargs)
        U_ = self.U[:,:self.n_components]
        X_transformed = np.dot(K, U_)
        return X_transformed


    def fit_transform(self, X, D, y = None):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            tranformed X_transformedmed
        '''
        self.fit(X, D, y)
        X_transformed = self.transform(X)
        if self.aug:
            X_transformed = np.concatenate((X_transformed, D), axis=1)
        return X_transformed