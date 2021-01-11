# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk, The University of Sheffield
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from ..utils.mmd import mmd_coef
from ..utils.multiclass import score2pred
from .base import SSLFramework
# =============================================================================
# Adaptation Regularisation Transfer Learning: ARTL
# Ref: Long, M., Wang, J., Ding, G., Pan, S.J. and Philip, S.Y., 2013. 
# Adaptation regularization: A general framework for transfer learning. 
# IEEE Transactions on Knowledge and Data Engineering, 26(5), pp.1076-1089.
# =============================================================================


def _init_artl(Xs, ys, Xt, yt=None, **kwargs):
    """[summary]

    Parameters
    ----------
    Xs : array-like
        Source data, shape (ns_samples, n_feautres)
    ys : array-like
        Source labels, shape (ns_samples,)
    Xt : array-like
        Target data, shape (nt_samples, n_feautres), the first ntl
        samples are labelled if yt is not None
    yt : array-like, optional
        Target label, shape (ntl_samples, ), by default None

    Returns
    -------
    X : array-like
        [description]
    y : array-like

    K : array-like

    M : array-like

    I : array-like

    """
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    n = ns + nt
    if yt is not None:
        X = np.concatenate([Xs, Xt], axis=0)
        y = np.concatenate([ys, yt])
    else:
        X = np.concatenate([Xs, Xt], axis=0)
        y = ys.copy()

    M = mmd_coef(ns, nt, ys, yt, kind='joint')
    K = pairwise_kernels(X, **kwargs)
    K[np.isnan(K)] = 0
    I = np.eye(n)

    return X, y, K, M, I


class ARSVM(SSLFramework):
    def __init__(self, C=1.0, kernel='linear', lambda_=1.0, gamma_=0.0, k_neighbour=5,
                 solver='osqp', manifold_metric='cosine', knn_mode='distance', **kwargs):
        """Adaptation Regularised Support Vector Machine

        Parameters
        ----------
        C : float, optional
            param for importance of slack variable, by default 1.0
        kernel : str, optional
            'rbf' | 'linear' | 'poly' , by default 'linear'
        lambda_ : float, optional
            MMD regulisation param, by default 1.0
        gamma_ : float, optional
            manifold regulisation param, by default 0.0
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation, 
            by default 5
        solver : str, optional
            solver to solve quadprog, osqp or cvxopt, by default 'osqp'
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each 
            sample point. The DistanceMetric class gives a list of available 
            metrics. By default 'cosine'.
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of 
            returned matrix: ‘connectivity’ will return the connectivity 
            matrix with ones and zeros, and ‘distance’ will return the 
            distances between neighbors according to the given metric.
        kwargs :
            kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = C
        self.gamma_ = gamma_
        self.solver = solver
        self.k_neighbour = k_neighbour
        # self.alpha = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)
        # self.scaler = StandardScaler()

    def fit(self, Xs, ys, Xt, yt=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_feautres)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_feautres), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        X, y, K, M, I = _init_artl(Xs, ys, Xt, yt, metric=self.kernel,
                                   filter_params=True, **self.kwargs)

        y_ = self._lb.fit_transform(y)

        if self.gamma_ != 0:
            L = self._lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode)
            Q_ = I + multi_dot([(self.lambda_ * M + self.gamma_ * L), K])
        else:
            Q_ = I + multi_dot([(self.lambda_ * M), K])

        self.coef_, self.support_ = self._solve_semi_dual(K, y_, Q_, self.C, self.solver)
        # if self._lb.y_type_ == 'binary':
        #     self.support_vectors_ = X[:nl, :][self.support_]
        #     self.n_support_ = self.support_vectors_.shape[0]
        # else:
        #     self.support_vectors_ = []
        #     self.n_support_ = []
        #     for i in range(y_.shape[1]):
        #         self.support_vectors_.append(X[:nl, :][self.support_[i]][-1])
        #         self.n_support_.append(self.support_vectors_[-1].shape[0])

        self._X = X
        self._y = y

        return self

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like
            shape (n_samples, n_feautres)

        Returns
        -------
        array-like
            decision scores, , shape (n_samples,) for binary classification, 
            (n_samples, n_class) for multi-class cases
        """
        check_is_fitted(self, '_X')
        check_is_fitted(self, '_y')
        # X_fit = self._X
        K = pairwise_kernels(X, self._X, metric=self.kernel, filter_params=True, **self.kwargs)

        return np.dot(K, self.coef_)  # +self.intercept_

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            shape (n_samples, n_feautres)

        Returns
        -------
        array-like
            predicted labels, , shape (n_samples, )
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, Xs, ys, Xt, yt=None):
        """Fit the model according to the given training data and then perform
            classification on samples in Xt.
        
        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_feautres)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_feautres), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        self.fit(Xs, ys, Xt, yt)

        return self.predict(self._X)


class ARRLS(SSLFramework):
    def __init__(self, kernel='linear', lambda_=1.0, gamma_=0.0, sigma_=1.0, 
                 k_neighbour=5, manifold_metric='cosine', knn_mode='distance', 
                 **kwargs):
        """Adaptation Regularised Least Square

        Parameters
        ----------
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        lambda_ : float, optional
            MMD regularisation param, by default 1.0
        gamma_ : float, optional
            manifold regularisation param, by default 0.0
        sigma_ : float, optional
            l2 regularisation param, by default 1.0
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation, 
            by default 5
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each 
            sample point. The DistanceMetric class gives a list of available 
            metrics. By default 'cosine'.
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of 
            returned matrix: ‘connectivity’ will return the connectivity 
            matrix with ones and zeros, and ‘distance’ will return the 
            distances between neighbors according to the given metric.
        kwargs: 
            kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.sigma_ = sigma_
        self.k_neighbour = k_neighbour
        # self.coef_ = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, Xs, ys, Xt, yt=None):
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_feautres)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_feautres), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        X, y, K, M, I = _init_artl(Xs, ys, Xt, yt, metric=self.kernel,
                                   filter_params=True, **self.kwargs)

        n = K.shap[0]
        nl = y.shape[0]

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            L = self._lapnorm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode,
                        metric=self.manifold_metric)
            Q_ = np.dot((J + self.lambda_ * M + self.gamma_ * L),
                        K) + self.sigma_ * I
        else:
            Q_ = np.dot((J + self.lambda_ * M), K) + self.sigma_ * I

        y_ = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q_, y_)

        self._X = X
        self._y = y

        return self

    def predict(self, X):
        """Perform classification on samples in X.
        
        Parameters:
        ----------
            X : array-like, 
                shape (n_samples, n_feautres)
        Returns
        -------
        array-like
            predicted labels, shape (n_samples)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == 'binary':
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
            X : array-like, 
                shape (n_samples, n_feautres)
        Returns
        -------
        array-like
            prediction scores, shape (n_samples)
        """
        K = pairwise_kernels(X, self._X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(K, self.coef_)  

    def fit_predict(self, Xs, ys, Xt, yt=None):
        """Fit the model according to the given training data and then perform
            classification on samples in Xt.

        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_feautres)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_feautres), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        self.fit(Xs, ys, Xt, yt)

        return self.predict(Xt)
