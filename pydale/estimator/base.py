import warnings
import numpy as np
from numpy.linalg import multi_dot, inv
import scipy.sparse as sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from cvxopt import matrix, solvers
import osqp


class BaseFramework(BaseEstimator, ClassifierMixin):
    """Semi-supervised Learning Framework
    """
    def __init__(self, kernel, **kwargs) -> None:
        super().__init__()
        self.kernel = kernel
        self.coef_ = None
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)
        self.kwargs = kwargs
        self.x = None

    @classmethod
    def _solve_semi_dual(cls, K, y, Q_, C, solver='osqp'):
        """[summary]

        Parameters
        ----------
        K : [type]
            [description]
        y : [type]
            [description]
        Q_ : [type]
            [description]
        C : [type]
            [description]
        solver : str, optional
            [description], by default 'osqp'

        Returns
        -------
        [type]
            [description]
        """
        if len(y.shape) == 1:
            coef_, support_ = cls._semi_binary_dual(K, y, Q_, C, solver)
            support_ = [support_]
        else:
            coef_ = []
            support_ = []
            for i in range(y.shape[1]):
                coef_i, support_i = cls._semi_binary_dual(K, y[:, i], Q_, C, solver)
                coef_.append(coef_i.reshape(-1, 1))
                support_.append(support_i)

            coef_ = np.concatenate(coef_, axis=1)

        return coef_, support_

    @classmethod
    def _semi_binary_dual(cls, K, y_, Q_, C, solver='osqp'):
        """solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b

        Parameters
        ----------
        K : [type]
            [description]
        y_ : [type]
            [description]
        Q_ : [type]
            [description]
        C : [type]
            [description]
        solver : str, optional
            [description], by default 'osqp'

        Returns
        -------
        [type]
            [description]
        """
        nl = y_.shape[0]
        n = K.shape[0]
        J = np.zeros((nl, n))
        J[:nl, :nl] = np.eye(nl)
        Q_inv = inv(Q_)
        Y = np.diag(y_.reshape(-1))
        Q = multi_dot([Y, J, K, Q_inv, J.T, Y])
        Q = Q.astype('float32')
        alpha = cls._quadprog(Q, y_, C, solver)
        coef_ = multi_dot([Q_inv, J.T, Y, alpha])
        support_ = np.where((alpha > 0) & (alpha < C))
        return coef_, support_

    @classmethod
    def _quadprog(cls, Q, y, C, solver='osqp'):
        """solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b

        Parameters
        ----------
        Q : [type]
            [description]
        y : [type]
            [description]
        C : [type]
            [description]
        solver : str, optional
            [description], by default 'osqp'

        Returns
        -------
        [type]
            [description]
        """
        # dual
        nl = y.shape[0]
        q = -1 * np.ones((nl, 1))

        if solver == 'cvxopt':
            G = np.zeros((2 * nl, nl))
            G[:nl, :] = -1 * np.eye(nl)
            G[nl:, :] = np.eye(nl)
            h = np.zeros((2 * nl, 1))
            h[nl:, :] = C / nl

            # convert numpy matrix to cvxopt matrix
            P = matrix(Q)
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)
            A = matrix(y.reshape(1, -1).astype('float64'))
            b = matrix(np.zeros(1).astype('float64'))

            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b)

            alpha = np.array(sol['x']).reshape(nl)

        elif solver == 'osqp':
            warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
            P = sparse.csc_matrix((nl, nl))
            P[:nl, :nl] = Q[:nl, :nl]
            G = sparse.vstack([sparse.eye(nl), y.reshape(1, -1)]).tocsc()
            l_ = np.zeros((nl + 1, 1))
            u = np.zeros(l_.shape)
            u[:nl, 0] = C

            prob = osqp.OSQP()
            prob.setup(P, q, G, l_, u, verbose=False)
            res = prob.solve()
            alpha = res.x

        else:
            raise ValueError('Invalid QP solver')

        return alpha

    @classmethod
    def _solve_semi_ls(cls, Q, y):
        """[summary]

        Parameters
        ----------
        Q : [type]
            [description]
        y : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        n = Q.shape[0]
        n_labeled = y.shape[0]
        Q_inv = inv(Q)
        if len(y.shape) == 1:
            y_ = np.zeros(n)
            y_[:n_labeled] = y[:]
        else:
            y_ = np.zeros((n, y.shape[1]))
            y_[:n_labeled, :] = y[:, :]
        return np.dot(Q_inv, y_)
    
    def predict(self, x):
        """Perform classification on samples in x.

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
            
        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        dec_scores = self.decision_function(x)
        # if self._lb.y_type_ == 'binary':
        #     y_pred = self._lb.inverse_transform(np.sign(dec_scores).reshape(-1, 1))
        # else:
        #     y_pred = self._lb.inverse_transform(dec_scores)
        y_pred = self._lb.inverse_transform(dec_scores)

        return y_pred

    def decision_function(self, x):
        """Evaluates the decision function for the samples in x

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
            
        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification, 
            (n_samples, n_class) for multi-class cases
        """
        check_is_fitted(self, 'x')
        krnl_x = pairwise_kernels(x, self.x, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(krnl_x, self.coef_) 
