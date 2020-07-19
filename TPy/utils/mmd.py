# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk, The University of Sheffield
# =============================================================================
import sys
import numpy as np


def mmd_coef(ns, nt, ys=None, yt=None, kind='marginal', mu=1):
    n = ns + nt
    e = np.zeros((n, 1))
    e[:ns, 0] = 1.0 / ns
    e[ns:, 0] = -1.0 / nt
    M = np.dot(e, e.T)

    if kind == 'joint' and ys is not None:
        class_all = np.unique(ys)
        if yt is not None and class_all.all() != np.unique(yt).all():
            sys.exit('Source and target domain should have the same labels')

        for c in class_all:
            es = np.zeros([ns, 1])
            es[np.where(ys == c)] = 1.0 / (np.where(ys == c)[0].shape[0])
            et = np.zeros([nt, 1])
            if yt is not None:
                et[np.where(yt == c)[0]] = -1.0 / np.where(yt == c)[0].shape[0]
            e = np.vstack((es, et))
            e[np.where(np.isinf(e))[0]] = 0
            M = M + mu * np.dot(e, e.T)

    return M