# =============================================================================
# author: Shuo Zhou, The University of Sheffield
# =============================================================================
import sys
import warnings
import numpy as np
from scipy.linalg import eig, sqrtm
import scipy.sparse as sparse
from numpy.linalg import multi_dot, inv, solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
# import cvxpy as cvx
# from cvxpy.error import SolverError
from cvxopt import matrix, solvers
import osqp

# =============================================================================
# Adaptation Regularisation Transfer Learning: ARTL
# Ref: Long, M., Wang, J., Ding, G., Pan, S.J. and Philip, S.Y., 2013. 
# Adaptation regularization: A general framework for transfer learning. 
# IEEE Transactions on Knowledge and Data Engineering, 26(5), pp.1076-1089.
# =============================================================================


def get_kernel(X, Y=None, kernel='linear', **kwargs):
    """
    Generate kernel matrix
    Parameters:
        X: X matrix (n1,d)
        Y: Y matrix (n2,d)
        kernel: 'linear'(default) | 'rbf' | 'poly'
    Return:
        Kernel matrix

    """

    return pairwise_kernels(X, Y=Y, metric=kernel,
                            filter_params=True, **kwargs)


def get_lapmat(X, n_neighbour=5, metric='cosine', mode='distance',
               normalise=True):
    n = X.shape[0]
    knn_graph = kneighbors_graph(X, n_neighbour, metric=metric,
                                 mode=mode).toarray()
    W = np.zeros((n, n))
    knn_idx = np.logical_or(knn_graph, knn_graph.T)
    if mode == 'distance':
        graph_kernel = pairwise_distances(X, metric=metric)
        W[knn_idx] = graph_kernel[knn_idx]
    else:
        W[knn_idx] = 1

    D = np.diag(np.sum(W, axis=1))
    if normalise:
        D_ = inv(sqrtm(D))
        lapmat = np.eye(n) - multi_dot([D_, W, D_])
    else:
        lapmat = D - W
    return lapmat


class ARSVM(BaseEstimator, TransformerMixin):
    def __init__(self, C=1, kernel='linear', lambda_=1, gamma_=0, k=5, solver='osqp', 
                 manifold_metric='cosine', knn_mode='distance', **kwargs):
        """
        Init function
        Parameters
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            lambda_: MMD regulization param
            gamma_: manifold regulization param
            solver: osqp (default), cvxopt
            kwargs: kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = C
        self.gamma_ = gamma_
        self.solver = solver
        self.n_neighbour = k
        self.alpha = None
        self.mode = knn_mode
        # self.scaler = StandardScaler()

    def fit(self, Xs, ys, Xtu, Xtl=None, yt=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xtu: Unlabelled target data,  array-like, shape (ntu_samples, n_feautres)
            Xtl: Labelled target data, array-like, shape (ntl_samples, n_feautres)
            yt: Target label, array-like, shape (ntl_samples, )
        """

        ns = Xs.shape[0]
        ntl = 0
        if Xtl is not None:
            ntl += Xtl.shape[0]
            X = np.concatenate([Xs, Xtl, Xtu], axis=0)
            y = np.concatenate([ys, yt])
        else:
            X = np.concatenate([Xs, Xtu], axis=0)
            y = ys.copy()

        n = X.shape[0]
        nt = ntl + Xtu.shape[0]
        nl = ns + ntl  # number of labelled data
        # X = self.scaler.fit_transform(X)

        e = np.zeros((n,1))
        e[:ns, 0] = 1.0 / ns
        e[ns:, 0] = -1.0 / nt
        M = np.dot(e, e.T)

        class_all = np.unique(ys)
        if yt is not None and class_all.all() != np.unique(yt).all():
            sys.exit('Source and target domain should have the same labels')

        for c in class_all:
            e1 = np.zeros([ns, 1])
            e1[np.where(ys == c)] = 1.0 / (np.where(ys == c)[0].shape[0])
            e2 = np.zeros([nt, 1])
            if yt is not None:
                e2[np.where(yt == c)[0]] = -1.0 / np.where(yt == c)[0].shape[0]
            e = np.vstack((e1, e2))
            e[np.where(np.isinf(e))[0]] = 0
            M = M + np.dot(e, e.T)

        I = np.eye(n)
        # H = I - 1. / n * np.ones((n, n))
        K = get_kernel(X, kernel=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        # dual
        Y = np.diag(y)
        J = np.zeros((nl, n))
        J[:nl, :nl] = np.eye(nl)
        if self.gamma_ != 0:
            L = get_lapmat(X, n_neighbour=self.n_neighbour)
            Q_ = self.C * I + multi_dot([(self.lambda_ * M + self.gamma_ * L), K])
        else:
            Q_ = self.C * I + multi_dot([(self.lambda_ * M), K])
        Q = multi_dot([Y, J, K, inv(Q_), J.T, Y])
        q = -1 * np.ones((nl, 1))

        if self.solver == 'cvxopt':
            G = np.zeros((2 * nl, nl))
            G[:nl, :] = -1 * np.eye(nl)
            G[nl:, :] = np.eye(nl)
            h = np.zeros((2 * nl, 1))
            h[nl:, :] = 1 / nl

            # convert numpy matrix to cvxopt matrix
            P = matrix(Q)
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)
            A = matrix(y.reshape(1, -1).astype('float64'))
            b = matrix(np.zeros(1).astype('float64'))

            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)
            self.alpha = np.array(sol['x']).reshape(nl)

        elif self.solver == 'osqp':
            warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
            P = sparse.csc_matrix((nl, nl))
            P[:nl, :nl] = Q[:nl, :nl]
            G = sparse.vstack([sparse.eye(nl), y.reshape(1, -1)]).tocsc()
            l = np.zeros((nl+1, 1))
            u = np.zeros(l.shape)
            u[:nl, 0] = 1 / nl

            prob = osqp.OSQP()
            prob.setup(P, q, G, l, u, verbose=False)
            res = prob.solve()
            self.alpha = res.x
        else:
            print('Invalid QP solvers')
            sys.exit()

        self.coef_ = multi_dot([inv(Q_), J.T, Y, self.alpha])
        self.support_ = np.where((self.alpha > 0) & (self.alpha < self.C))
        self.support_vectors_ = X[:nl, :][self.support_, :]
        self.n_support_ = self.support_vectors_.shape[0]
        # K_train = get_kernel(X_train, X, kernel=self.kernel, **self.kwargs)
        # self.intercept_ = np.mean(y[self.support_] - y[self.support_] *
        #                           np.dot(K_train[self.support_], self.coef_))/self.n_support_

# =============================================================================
#         beta = cvx.Variable(shape = (2 * n, 1))
#         objective = cvx.Minimize(cvx.quad_form(beta, P) + q.T * beta)
#         constraints = [G * beta <= h]
#         prob = cvx.Problem(objective, constraints)
#         try:
#             prob.solve()
#         except SolverError:
#             prob.solve(solver = 'SCS')
#         
#         self.coef_ = beta.value[:n]
# =============================================================================
        
#        a = np.dot(W + self.gamma * multi_dot([H, Ka, H]), self.lambda_*I)
#        b = np.dot(y, W)
#        beta = solve(a, b)

        self.X = X
        self.y = y
        
        return self

    def decision_function(self, X):
        # check_is_fitted(self, 'X')
        # check_is_fitted(self, 'y')
        # X_fit = self.X
        K = get_kernel(X, self.X, kernel=self.kernel, **self.kwargs)
        # K = get_kernel(self.scaler.transform(X), X_fit,
        #                kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples, )
        """
        
        # return np.sign(self.decision_function(self.scaler.transform(X)))
        return np.sign(self.decision_function(X))

    def fit_predict(self, Xs, ys, Xtl=None, Xtu=None, yt=None):
        """
        solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            Xtl: Labelled target data, array-like, shape (ntl_samples, n_feautres)
            Xtu: Unlabelled target data,  array-like, shape (ntu_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            yt: Target label, array-like, shape (ntl_samples, )
        """
        self.fit(Xs, Xtl, Xtu, ys, yt)
        y_pred = self.predict(Xtu)
        return y_pred


class ARRLS(BaseEstimator, TransformerMixin):
    def __init__(self, kernel='linear', lambda_=1, gamma_=0, sigma_=1, k=5, 
                 manifold_metric='cosine', knn_mode='distance', **kwargs):
        """
        Init function
        Parameters
            kernel: 'rbf' | 'linear' | 'poly' (default is 'linear')
            lambda_: MMD regularisation param
            gamma_: manifold regularisation param
            sigma_: l2 regularisation param
            solver: osqp (default), cvxopt
            kwargs: kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.sigma_ = sigma_
        self.n_neighbour = k
        self.classes = None
        self.coef_ = None
        self.mode = knn_mode
        self.manifold_metric = manifold_metric

    def fit(self, Xs, ys, Xtu, Xtl=None, yt=None):
        """
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xtu: Unlabelled target data,  array-like, shape (ntu_samples, n_feautres)
            Xtl: Labelled target data, array-like, shape (ntl_samples, n_feautres)
            yt: Target label, array-like, shape (ntl_samples, )
        """
        ns = Xs.shape[0]
        ntl = 0
        if Xtl is not None:
            ntl += Xtl.shape[0]
            X = np.concatenate([Xs, Xtl, Xtu], axis=0)
            y = np.concatenate([ys, yt])
        else:
            X = np.concatenate([Xs, Xtu], axis=0)
            y = ys.copy()

        n = X.shape[0]
        nt = ntl + Xtu.shape[0]
        nl = ns + ntl  # number of labelled data
        # X = self.scaler.fit_transform(X)
        self.classes = np.unique(y)

        e = np.zeros((n, 1))
        e[:ns, 0] = 1.0 / ns
        e[ns:, 0] = -1.0 / nt
        M = np.dot(e, e.T)

        class_all = np.unique(ys)
        if yt is not None and class_all.all() != np.unique(yt).all():
            sys.exit('Source and target domain should have the same labels')

        for c in class_all:
            e1 = np.zeros([ns, 1])
            e2 = np.zeros([nt, 1])
            e1[np.where(ys == c)] = 1.0 / (np.where(ys == c)[0].shape[0])
            if yt is not None:
                e2[np.where(yt == c)[0]] = -1.0 / np.where(yt == c)[0].shape[0]
            e = np.vstack((e1, e2))
            e[np.where(np.isinf(e))[0]] = 0
            M = M + np.dot(e, e.T)

        I = np.eye(n)
        # H = I - 1. / n * np.ones((n, n))
        K = get_kernel(X, kernel=self.kernel, **self.kwargs)
        K[np.isnan(K)] = 0

        E = np.zeros((n, n))
        E[:nl, :nl] = np.eye(nl)

        y_ = np.zeros(n)
        y_[:nl] = y[:]

        if self.gamma_ != 0:
            L = get_lapmat(X, n_neighbour=self.n_neighbour, mode=self.mode,
                           metric=self.manifold_metric)
            Q_ = np.dot((E + self.lambda_ * M + self.gamma_ * L),
                        K) + self.sigma_ * I
        else:
            Q_ = np.dot((E + self.lambda_ * M), K) + self.sigma_ * I
        Q_inv = inv(Q_)
        self.coef_ = multi_dot([Q_inv, E, y_])

        self.X = X
        self.y = y

        return self

    def predict(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            predicted labels, array-like, shape (n_samples)
        """
        dec = self.decision_function(X)
        if self.classes.shape[0] == 2:
            y_pred = np.sign(dec)
        else:
            n_test = X.shape[0]
            y_pred = np.zeros(n_test)
            dec_sort = np.argsort(dec, axis=1)[:, ::-1]
            for i in range(n_test):
                y_pred[i] = self.classes[dec_sort[i, 0]]

        return y_pred

    def decision_function(self, X):
        """
        Parameters:
            X: array-like, shape (n_samples, n_feautres)
        Return:
            prediction scores, array-like, shape (n_samples)
        """
        # check_is_fitted(self, 'X')
        # check_is_fitted(self, 'y')
        # K = get_kernel(self.scaler.transform(X), self.X,
        #                kernel=self.kernel, **self.kwargs)
        K = get_kernel(X, self.X, kernel=self.kernel, **self.kwargs)
        return np.dot(K, self.coef_)  

    def fit_predict(self, Xs, ys, Xtu, Xtl=None, yt=None):
        """
        Parameters:
            Xs: Source data, array-like, shape (ns_samples, n_feautres)
            ys: Source label, array-like, shape (ns_samples, )
            Xtu: Unlabelled target data,  array-like, shape (ntu_samples, n_feautres)
            Xtl: Labelled target data, array-like, shape (ntl_samples, n_feautres)
            yt: Target label, array-like, shape (ntl_samples, )
        """
        self.fit(Xs, ys, Xtu, Xtl, yt)
        return self.predict(Xtu)
