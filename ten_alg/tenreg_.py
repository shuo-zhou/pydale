import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
import tensorly as tl
from tensorly.base import unfold


def remurs(X, y, W=None, mu1=1, mu2=1, n_iter=1000, lr=0.0001):
    if W is None:
        W = torch.tensor(rng.random_sample((X.shape[:-1])), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=lr)
    lmbda = lambda epoch: 0.95
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    n_mode = len(W.shape)
    loss_prev = float('Inf')
    for epoch in range(n_iter):
        optimizer.zero_grad()
        pred = inner(W, X, n_modes=n_mode)
        criterion = nn.MSELoss(reduction='sum')
        loss = criterion(y, pred) + mu1 * torch.norm(W, p=1)
        
        for j in range(n_mode):
            loss = loss + mu2 * torch.norm(unfold(W, mode=j), p='nuc')
            
        if epoch % 10 == 0:
            print("Epoch %s" % epoch)
            print('Overall loss:', loss.item())
            if loss.item() < loss_prev or epoch < 50:
                loss_prev = loss.item()
            else:
                break
                
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return W


def HSIC(X, Y):
    n = X.shape[0]
    if len(X.shape) == 1:
        X = X.view(X.shape[0], 1)
    K = torch.matmul(X, torch.transpose(X, 0, 1))
    Y = torch.from_numpy(Y)
    L = torch.matmul(Y, torch.transpose(Y, 0, 1))
    H = torch.from_numpy(np.eye(n) - 1. / n * np.ones((n, n)))
    
    return torch.trace(torch.matmul(torch.matmul(torch.matmul(K, H), L), H)) # /((n-1)**2)


def TMR(X, y, train_idx, D, W=None, mu1=1, mu2=1, mu3=1, n_iter=1000, lr=0.001):
    if W is None:
        W = torch.tensor(rng.random_sample((X.shape[:-1])), requires_grad=True)
    optimizer = torch.optim.Adam([W], lr=lr)

    lmbda = lambda epoch: 0.95
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    n_mode = len(W.shape)
    loss_prev = float('Inf')
    for epoch in range(n_iter):
        optimizer.zero_grad()
        dec_train = inner(W, X[..., train_idx], n_modes=n_mode)
        criterion = nn.MSELoss(reduction='sum')
        loss = criterion(y, dec_train) + mu1 * torch.norm(W, p=1)

        for j in range(n_mode):
            loss = loss + mu2 * torch.norm(unfold(W, mode=j), p='nuc')
        
        dec_ = inner(W, X, n_modes=n_mode)
        hsic = HSIC(dec_, D)
        loss = loss + mu3 * hsic
        
        if epoch % 10 == 0:
            print("Epoch %s" % epoch)
            print('Overall loss:', loss.item())
            if loss.item() < loss_prev or epoch < 50:
                loss_prev = loss.item()
            else:
                break
                
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return W
