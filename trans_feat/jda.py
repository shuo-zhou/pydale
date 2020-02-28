# =============================================================================
# author: Shuo Zhou, The University of Sheffield
# =============================================================================
import sys
import numpy as np
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
#from sklearn.preprocessing import StandardScaler
# =============================================================================
# Joint Distribution Adaptation: JDA
# Ref: Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu,
# Transfer Feature Learning with Joint Distribution Adaptation, IEEE 
# International Conference on Computer Vision (ICCV), 2013.
# =============================================================================

class JDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel = 'linear', lambda_=1, **kwargs):
        '''
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel_type: [‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, 
            ‘cosine’] (default is 'linear')
            **kwargs: kernel param
            lambda_: regulization param
        '''
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
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
            Kernel matrix
        '''

        return pairwise_kernels(X, Y=Y, metric = self.kernel, 
                                filter_params = True, **self.kwargs)

    def fit(self, Xs, Xt, ys, yt):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
            ys: Labels of source domain samples, shape (n_samples,)
            yt: Labels of source domain samples, shape (n_samples,)
        '''
        X = np.vstack((Xs, Xt))
#        self.scaler = StandardScaler()
#        self.scaler.fit(X)
#        X = self.scaler.transform(X)
        #X = np.dot(X.T, np.diag(1.0 / np.sqrt(np.sum(np.square(X), axis = 1)))).T 
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        n = ns + nt
        class_all = np.unique(ys)
        if class_all.all() != np.unique(yt).all():
            sys.exit('Source and target domain should have the same labels')
        C = len(class_all)
    
        # Construct MMD kernel weight matrix
        L = self.get_L(ns, nt) * C
    
        # Within class MMD kernel weight matrix
        if len(yt) != 0 and len(yt) == nt:
            for c in class_all:
                e1 = np.zeros([ns, 1])
                e2 = np.zeros([nt, 1])
                e1[np.where(ys == c)] = 1.0 / (np.where(ys == c)[0].shape[0])
                e2[np.where(yt == c)[0]] = -1.0 / np.where(yt == c)[0].shape[0]
                e = np.vstack((e1, e2))
                e[np.where(np.isinf(e))[0]] = 0
                L = L + np.dot(e, e.T)
        else:
            sys.exit('Target domain data and label should have the same size!')
    
        divider = np.sqrt(np.sum(np.diag(np.dot(L.T, L))))
        L = L / divider
        
        # Construct kernel matrix
        K = self.get_kernel(X, None)
    
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

        U[:,:] = eig_vecs[:, idx_sorted]
        self.U = np.asarray(U, dtype = np.float)
        self.K = K
        self.Xs = Xs
        self.Xt = Xt
        self.nt = nt
        self.ns = ns
        return self
    
    def transform(self, X):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return: 
            tranformed data
        '''
#        X = self.scaler.transform(X)
        check_is_fitted(self, 'Xs')
        check_is_fitted(self, 'Xt')
        X_fit = np.vstack((self.Xs, self.Xt))
        K = self.get_kernel(X, X_fit)
        U_ = self.U[:,:self.n_components]
        X_transformed = np.dot(K, U_)
        return X_transformed
    
    def fit_transform(self, Xs, Xt, ys, yt):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
            ys: Labels of source domain samples, shape (n_samples,)
            yt: Labels of source domain samples, shape (n_samples,)
        Return: 
            tranformed Xs_transformed, Xt_transformed
        '''
        self.fit(Xs, Xt, ys, yt)
        Xs_transformed = self.transform(Xs)
        Xt_transformed = self.transform(Xt)
        return Xs_transformed, Xt_transformed
