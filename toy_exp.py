#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:19:22 2018

@author: shuoz
"""

import matplotlib.pylab as plt
import numpy as np
import toy_data
import toydata
from feature_adaptation.jda import JDA
from feature_adaptation.vda import VDA
from feature_adaptation.tca import TCA
#from feature_adap.da_tool.tca import TCA
from jtda import JTDA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import seaborn as sns

def rev_label(y):
    rev_y = np.zeros(len(y))
    uni_label = np.unique(y)
    rev_y[y==uni_label[0]] = uni_label[1]
    rev_y[y==uni_label[1]] = uni_label[0]
    return rev_y

def evaluate(pred, y):
    pred_ = rev_label(pred)
    if accuracy_score(y, pred)>accuracy_score(y, pred_):
        print (accuracy_score(y, pred))
    else:
        print (accuracy_score(y, pred_))


random_state = 10

#Xs, ys, Xt, yt = toy_data.get_toydata(n=1000, dim=3)
Xs, ys, Xt, yt = toydata.get_toydata(n_features = 3, mismatch = 'joint')
y_all = np.hstack((ys, yt))
#ys = rev_label(ys)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], marker='o', c=ys, label = 'source')
ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], marker='+', c=yt, label = 'target')
plt.legend()
plt.savefig("data.eps",format="eps")
plt.show()
# 2D
#plt.scatter(Xs[:, 0], Xs[:, 1], marker='o', c=ys, label = 'source')
#plt.scatter(Xt[:, 0], Xt[:, 1], marker='+', c=yt, label = 'target')
#plt.legend()
#plt.savefig("data.eps",format="eps")
#plt.show()

#
##PCA
pca = PCA(n_components = 2)
pca.fit(np.vstack((Xs, Xt)))
Xs_pc = pca.transform(Xs)
Xt_pc = pca.transform(Xt)
plt.scatter(Xs_pc[:, 0], Xs_pc[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(Xt_pc[:, 0], Xt_pc[:, 1], marker='+', c=yt, label = 'target')
#sns.distplot(Xs_pc)
#sns.distplot(Xt_pc)

plt.legend()
plt.savefig("pca.eps", format = "eps")
plt.show()

#
ns = Xs.shape[0]
nt = Xt.shape[0]

#TCA
my_tca = TCA(2, kernel_type='linear', lambda_ = 10)
#my_tca = TCA(dim=1, kerneltype='linear', mu = 0.1)
Xtcs, Xtct = my_tca.fit_transform(Xs, Xt)

plt.scatter(Xtcs[:, 0], Xtcs[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(Xtct[:, 0], Xtct[:, 1], marker='+', c=yt, label = 'target')
#sns.distplot(Xtcs)
#sns.distplot(Xtct)

plt.legend()
plt.savefig("tca.eps", format="eps")
plt.show()

clf = SVC(kernel='linear')
y_pred = KMeans(n_clusters=2,
                random_state=random_state).fit_predict(np.vstack((Xtcs, Xtct)))
evaluate(y_pred, y_all)
clf.fit(Xtcs, ys)
print(accuracy_score(yt, clf.predict(Xtct)))


my_jda = JDA(2, kernel_type='linear', lambda_ = 1)
ZZs, ZZt = my_jda.fit_transform(Xs, Xt, ys, yt)

plt.scatter(ZZs[:, 0], ZZs[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(ZZt[:, 0], ZZt[:, 1], marker='+', c=yt, label = 'target')

#sns.distplot(ZZs)
#sns.distplot(ZZt)

plt.legend()
plt.savefig("jda.eps", format="eps")
plt.show()

y_pred = KMeans(n_clusters=2,
                random_state=random_state).fit_predict(np.vstack((ZZs, ZZt)))
evaluate(y_pred, y_all)
clf.fit(ZZs, ys)
print(accuracy_score(yt, clf.predict(ZZt)))


#ns = Xs.shape[0]
#nt = Xt.shape[0]
my_vda = VDA(2, kernel_type='linear', lambda_ = 1)
Zs, Zt = my_vda.fit_transform(Xs, Xt, ys, yt)
plt.scatter(Zs[:, 0], Zs[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(Zt[:, 0], Zt[:, 1], marker='+', c=yt, label = 'target')
plt.legend()
plt.savefig("vda.eps", format="eps")
plt.show()

y_pred = KMeans(n_clusters=2,
                random_state=random_state).fit_predict(np.vstack((Zs, Zt)))
evaluate(y_pred, y_all)
clf.fit(Zs, ys)
print(accuracy_score(yt, clf.predict(Zt)))

my_jtda = JTDA(2, kernel_type='linear', lambda_ = 1)
Zs, Zt = my_jtda.fit_transform(Xs, Xt, ys, yt)
plt.scatter(Zs[:, 0], Zs[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(Zt[:, 0], Zt[:, 1], marker='+', c=yt, label = 'target')
plt.legend()
plt.savefig("jtda.eps", format="eps")
plt.show()

y_pred = KMeans(n_clusters=2,
                random_state=random_state).fit_predict(np.vstack((Zs, Zt)))
evaluate(y_pred, y_all)
clf.fit(Zs, ys)
print(accuracy_score(yt, clf.predict(Zt)))
#
#y_pred = KMeans(n_clusters=2,
#                random_state=random_state).fit_predict(np.vstack((Zs, Zt)))
#evaluate(y_pred, y_all)
#
#clf.fit(Zs, ys)
#print(accuracy_score(yt, clf.predict(Zt)))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#Xs, ys, Xt, yt = get_toydata(n=100, mismatch = 'marginal')
#ax.scatter(Xs[:, 0], Xs[:, 1], Xs[:, 2], marker='o', c=ys, label = 'source')
#ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], marker='+', c=yt, label = 'target')
#plt.show()
#
#ns = Xs.shape[0]
#nt = Xt.shape[0]
#Z, A, phi = JTDA(Xs, Xt, ys, yt, 3, lmbda=1)
#
#Zs = Z[:ns, :]
#Zt = Z[ns:, :]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(Zs[:, 0], Zs[:, 1], Zs[:, 2], marker='o', c=ys, label = 'source')
#ax.scatter(Zt[:, 0], Zt[:, 1], Zt[:, 2], marker='+', c=yt, label = 'target')
#plt.show()

from model_adaptation.cdsvm import CDSVM

src_clf = SVC(kernel='linear')
src_clf.fit(Xtcs, ys)
cdsvm = CDSVM(src_clf.support_vectors_, ys[src_clf.support_])
cdsvm.fit(Xtct, yt)

svm =  SVC(kernel='linear')
svm.fit(np.vstack((Xtcs, Xtct)), np.hstack((ys, yt)))

# get the separating hyperplane
w = src_clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 28)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.scatter(Xtcs[:, 0], Xtcs[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(Xtct[:, 0], Xtct[:, 1], marker='+', c=yt, label = 'target')
plt.plot(xx, yy, 'k-')
plt.legend()
plt.show()

w = cdsvm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-50, 70)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.scatter(Xtcs[:, 0], Xtcs[:, 1], marker='o', c=ys, label = 'source')
plt.scatter(Xtct[:, 0], Xtct[:, 1], marker='+', c=yt, label = 'target')
plt.plot(xx, yy, 'k-')
plt.legend()
plt.show()