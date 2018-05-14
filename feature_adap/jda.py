# =============================================================================
# author: Shuo Zhou, The University of Sheffield
# =============================================================================
import numpy as np
import sys
import scipy.linalg 
from sklearn.metrics.pairwise import kernel_metrics

# =============================================================================
# Joint Distribution Adaptation: JDA
# Ref: Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu,
# Transfer Feature Learning with Joint Distribution Adaptation, IEEE 
# International Conference on Computer Vision (ICCV), 2013.
# =============================================================================

class JDA:
    def __init__(self, n_components, kernel_type='linear', lambda_=1, **kwargs):
        '''
        Init function
        Parameters
            n_components: n_componentss after tca (n_components <= d)
            kernel_type: 'rbf' | 'linear' | 'poly' (default is 'linear')
            **kwargs: kernel param
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

    def fit(self, Xs, Xt, ys, yt):
        '''
        Parameters:
            Xs: Source domain data, array-like, shape (n_samples, n_feautres)
            Xt: Target domain data, array-like, shape (n_samples, n_feautres)
            ys: Labels of source domain samples, shape (n_samples,)
            yt: Labels of source domain samples, shape (n_samples,)
        '''
        X = np.vstack((Xs, Xt))
        X = np.dot(X, np.diag(1.0 / np.sqrt(np.sum(np.square(X), axis = 1)))) 
        n, m = X.shape
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        class_all = np.unique(ys)
        if class_all != np.unique(yt):
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
        eig_values, eig_vecs = scipy.linalg.eig(obj, st)

        idx_sorted = eig_values.argsort()[::-1]
        W = eig_vecs[:, idx_sorted]
        W = W.reshape((W.shape[0], W.shape[1]))
        W = np.asarray(W, dtype = np.float)
        self.components_ = np.dot(X.T, W[:, :self.n_components])
        self.components_ = self.components_.T
    
    def transform(self, X):
        '''
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return: 
            tranformed data
        '''
        return np.dot(X, self.components_.T)
    
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
