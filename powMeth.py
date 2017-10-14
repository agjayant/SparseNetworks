import numpy as np
import random, math
from myTensorUtil import *

def getM1(X, y):
    d = np.shape(X)[1]
    M1 = np.zeros(d)
    for iter in range(np.shape(X)[0]):
        M1 += y[iter]*X[iter]
    return M1/len(y)

def getM2(X, y):
    d = np.shape(X)[1]
    M2 = np.zeros([d,d])
    for iter in range(np.shape(X)[0]):
        M2 += y[iter]*(np.outer(X[iter], X[iter]) - np.identity(d))
    return M2/len(y)

def getM3(X, y):
    d = np.shape(X)[1]
    M3 = np.zeros([d,d,d])
    for iter in range(np.shape(X)[0]):
        M3 += y[iter]*(outer3I(X[iter]) - specOuterI(X[iter]) )
    return M3/len(y)

def getM4(X, y):
    d = np.shape(X)[1]
    M4 = np.zeros([d,d,d,d])
    for iter in range(np.shape(X)[0]):
        M4 += y[iter]*(outer4I(X[iter]) - specOuterMat(np.outer(X[iter], X[iter])) + specOuterMat(np.identity(d)) )
    return M4/len(y)


def getP2V(V, X, y, k):
    d = np.shape(X)[1]
    P2V = np.zeros((d,k))
    for i in range(len(X)):
        P2V += y[i]*(np.dot(np.transpose([X[i]]), np.dot([X[i]], V) ) - V)
    return P2V/np.shape(X)[0]

def getP2v(v, X, y):
    d = np.shape(X)[1]
    P2v = np.zeros((d,1))
    for i in range(len(X)):
        P2v += y[i]*(np.transpose([X[i]])*np.dot([X[i]], np.transpose([v]))  - np.transpose([v]))
    return P2v/np.shape(X)[0]

def topk(eigenV, k):
    sortList = []
    for i in range(2):
        for j in range(k):
            sortList.append([eigenV[i,j], (i,j)])
    sortList.sort(reverse=True)
    k1 = 0
    k2 = 0
    pi1 = {}
    pi2 = {}
    for i in range(k):
        if sortList[i][1][0] == 0:
            pi1[k1] = sortList[i][1][1]
            k1 += 1
        else:
            pi2[k2] = sortList[i][1][1]
            k2 += 1
    return pi1, pi2, k1, k2

def powMeth( k, X, y):
#     C = 3*np.linalg.norm(P2)
    C = 3*np.linalg.norm(getM2(X,y))
    T = 10000
    d =  np.shape(X)[1]
    V1 = np.random.normal(0.0, 1.0 , (d,k))
    V2 = np.random.normal(0.0, 1.0 , (d,k))

    for i in range(T):
        P2V1 = getP2V(V1, X, y, k)
        P2V2 = getP2V(V2, X, y, k)
        V1, temp = np.linalg.qr(C*V1 + P2V1)
        V2, temp = np.linalg.qr(C*V2 - P2V2)

    eigenV = np.zeros((2,k))
    for i in range(k):
        eigenV[0,i] = abs( np.dot(np.transpose(V1[:,i]) , getP2v(V1[:,i], X, y) ) )
    for i in range(k):
        eigenV[1,i] = abs(np.dot(np.transpose(V2[:,i]) , getP2v(V2[:,i], X, y) ))

    pi1, pi2, k1, k2 = topk(eigenV, k)

    V1_new = np.zeros((d,k1))
    V2_new = np.zeros((d,k2))

    for i in range(k1):
        V1_new[:,i] = V1[:,pi1[i]]
    for i in range(k2):
        V2_new[:,i] = V2[:,pi2[i]]
    V2_new, temp = np.linalg.qr(np.dot(np.identity(d)-np.dot(V1_new, np.transpose(V1_new)), V2_new))
    return np.concatenate((V1_new, V2_new), axis=1)

