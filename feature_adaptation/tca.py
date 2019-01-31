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
# Transfer Component Analysis: TCA
# Ref: S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via 
# Transfer Component Analysis," in IEEE Transactions on Neural Networks, 
# vol. 22, no. 2, pp. 199-210, Feb. 2011.
# =============================================================================

def get_L(ns, nt):
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
      
def get_lapmat(X, k = 5):
    n = X.shape[0]
    knn_graph = kneighbors_graph(X, n_neighbors = k).toarray()
    knn_mat = np.zeros((n, n))
    knn_mat[np.logical_or(knn_graph, knn_graph.T)] = 1
    D = np.diag(np.sum(knn_mat, axis = 1))
    return D - knn_mat

class TCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel='linear', lambda_=1, mu = 1, gamma = 0.5, k = 5, **kwargs):
        '''
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            lambda_: regulization param
            mu: KNN graph param
            k: number of nearest neighbour for KNN graph
            gamma: label dependence param
        '''
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_ 
        self.mu = mu
        self.gamma = gamma
        self.k = k

    def fit(self, Xs, Xt, ys = None, yt = None, **kwargs):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (ns_samples, n_feautres)
            Xt: Target domain data, array-like, shape (nt_samples, n_feautres)
            ys: Source domain labels, array-like, shape (ns_samples,)
            yt: Target domain labels, array-like, shape (nt_samples,)
        Note:
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.
        '''
        self.ns = Xs.shape[0]
        self.nt = Xt.shape[0]
        n = self.ns + self.nt
        X = np.vstack((Xs, Xt))
        L = get_L(self.ns, self.nt)
        L[np.isnan(L)] = 0
        K = get_kernel(X, kernel = self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0
        
        I = np.eye(n)
        H = I - 1. / n * np.ones((n, n))

        if ys is not None and yt is not None:
            y = np.concatenate((ys, yt))    
            y = y.reshape((n,1))
            Kyy = self.gamma * np.dot(y, y.T) + (1-self.gamma) * I
            Lap_ = get_lapmat(K, k =self.k)
            obj = multi_dot([K, (L+ self.mu * Lap_), K.T]) + self.lambda_ * I
            st = multi_dot([K, H, Kyy, H, K.T])
        #obj = np.trace(np.dot(K,L))            
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
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            tranformed data
        '''
        check_is_fitted(self, 'Xs')
        check_is_fitted(self, 'Xt')
        X_fit = np.vstack((self.Xs, self.Xt))
        K = get_kernel(X, X_fit, kernel = self.kernel, **self.kwargs)
        U_ = self.U[:,:self.n_components]
        X_transformed = np.dot(K, U_)
        return X_transformed


    def fit_transform(self, Xs, Xt, ys=None, yt=None):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
        Return:
            tranformed Xs_transformed, Xt_transformed
        '''
        self.fit(Xs, Xt, ys, yt)
        Xs_transformed = self.transform(Xs)
        Xt_transformed = self.transform(Xt)
        return Xs_transformed, Xt_transformed