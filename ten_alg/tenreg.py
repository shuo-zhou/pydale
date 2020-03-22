import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
import tensorly as tl
from tensorly.base import unfold
from tensorly.tenalg import multi_mode_dot, inner
from sklearn.base import BaseEstimator, TransformerMixin


class REMURS(BaseEstimator, TransformerMixin):
    """
    Ref: Song, X. and Lu, H., 2017, February. Multilinear regression for 
    embedded feature selection with application to fmri analysis. In 
    Thirty-First AAAI Conference on Artificial Intelligence (AAAI2017).
    """
    def __init__(self, mu1=1, mu2=1, n_iter=1000, lr=0.0001, lmbda=0.95):
        self.mu1 = mu1
        self.mu2 = mu2
        self.n_iter = n_iter
        self.lr = lr
        self.lmbda = lmbda
        self.W = None
        self.loss_ = []
        tl.set_backend('pytorch')

    def fit(self, X, y):
        W = torch.zeros((X.shape[:-1]), dtype=torch.double, requires_grad=True)
        optimizer = torch.optim.Adam([W], lr=self.lr)
        lmbda = lambda epoch: self.lmbda
        scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
        n_mode = len(W.shape)
        loss_prev = float('Inf')
        self.loss_ = []

        for epoch in range(self.n_iter):
            optimizer.zero_grad()
            pred = inner(W, X, n_modes=n_mode)
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(y, pred) + self.mu1 * torch.norm(W, p=1)
            for j in range(n_mode):
                loss = loss + self.mu2 * torch.norm(unfold(W, mode=j), p='nuc')

            if epoch % 10 == 0:
                # print("Epoch %s" % epoch)
                # print('Overall loss:', loss.item())
                diff = loss.item() - loss_prev
                if  (diff < 0 and np.absolute(diff)/loss_prev > 0.01) or epoch < 30:
                    loss_prev = loss.item()
                else:
                    break

            loss.backward()
            optimizer.step()
            scheduler.step()

            self.loss_.append(loss.item())

        self.W = W

    def decision_fuction(self, X):
        n_mode = len(X.shape[:-1])
        return inner(self.W, X, n_modes=n_mode)

    def predict(self, X):
        dec = self.decision_fuction(X)
        return torch.sign(dec)


def HSIC(X, Y):
    n = X.shape[0]
    if len(X.shape) == 1:
        X = X.view(X.shape[0], 1)
    K = torch.matmul(X, torch.transpose(X, 0, 1))
    Y = torch.from_numpy(Y)
    L = torch.matmul(Y, torch.transpose(Y, 0, 1))
    H = torch.from_numpy(np.eye(n) - 1. / n * np.ones((n, n)))
    
    return torch.trace(torch.matmul(torch.matmul(torch.matmul(K, H), L), H)) # /((n-1)**2)


class TMR(BaseEstimator, TransformerMixin):
    def __init__(self, mu1=1, mu2=1, mu3=1, n_iter=1000, lr=0.0001, lmbda=0.95):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.n_iter = n_iter
        self.lr = lr
        self.lmbda = lmbda
        self.W = None
        self.loss_ = []
        tl.set_backend('pytorch')

    def fit(self, X, y, train_idx, D):
        W = torch.zeros((X.shape[:-1]), dtype=torch.double, requires_grad=True)
        optimizer = torch.optim.Adam([W], lr=self.lr)
        lmbda = lambda epoch: self.lmbda
        scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
        n_mode = len(W.shape)
        loss_prev = float('Inf')
        self.loss_ = []

        for epoch in range(self.n_iter):
            optimizer.zero_grad()
            dec_train = inner(W, X[..., train_idx], n_modes=n_mode)
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(y, dec_train) + self.mu1 * torch.norm(W, p=1)
            for j in range(n_mode):
                loss = loss + self.mu2 * torch.norm(unfold(W, mode=j), p='nuc')

            dec_ = inner(W, X, n_modes=n_mode)
            hsic = HSIC(dec_, D)
            loss = loss + self.mu3 * hsic

            if epoch % 10 == 0:
                # print("Epoch %s" % epoch)
                # print('Overall loss:', loss.item())
                diff = loss.item() - loss_prev
                if  (diff < 0 and np.absolute(diff)/loss_prev > 0.01) or epoch < 30:
                    loss_prev = loss.item()
                else:
                    break

            loss.backward()
            optimizer.step()
            scheduler.step()

            self.loss_.append(loss.item())

        self.W = W

    def decision_fuction(self, X):
        n_mode = len(X.shape[:-1])
        return inner(self.W, X, n_modes=n_mode)

    def predict(self, X):
        dec = self.decision_fuction(X)
        return torch.sign(dec)
