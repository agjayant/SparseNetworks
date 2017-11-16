import numpy as np
import math
import numpy as np
import random, math


## Activation Function
def phi_s(h):
        return h**2 if h > 0 else 0
phi = np.vectorize(phi_s)


## Moments
def gamma(j,sigma):
    estm = 0
    for i in range(10000):
        z = np.random.normal(0.0, 1.0)
        estm += phi_s(z*sigma)*(z**j)
    return estm/10000


def generateWeights(d = 10, k = 5, s_gt = 0.15, kappa = 2 ):
    gauss_mat_u = np.random.normal(0.0, 1.0 , (d,k))
    gauss_mat_v = np.random.normal(0.0, 1.0 , (k,k))

    U, temp = np.linalg.qr(gauss_mat_u)
    V, temp = np.linalg.qr(gauss_mat_v)

    # U, V = np.linalg.qr(gauss_mat_u)
    diag = []
    v_gt = []
    v_choice = [1,-1]
    for iter in range(k):
        diag.append(1+1.*iter*(kappa-1)/(k-1))
        v_gt.append(random.choice(v_choice))

    Sigma = np.diag(diag)
    W_gt = np.dot(np.dot(U, Sigma), np.transpose(V))
    v_gt = np.asarray(v_gt)

    for i in range(d):
        for j in range(k):
            this_norm = np.linalg.norm(W_gt[i,j])
            if this_norm <= s_gt:
                W_gt[i,j] = 0.0

    m = np.zeros((4,k))
    for i in range(k):
        m[0,i] = gamma(1,np.linalg.norm(W_gt[:,i]))
        m[1,i] = gamma(2,np.linalg.norm(W_gt[:,i])) - gamma(0,np.linalg.norm(W_gt[:,i]))
        m[2,i] = gamma(3,np.linalg.norm(W_gt[:,i])) - 3*gamma(1,np.linalg.norm(W_gt[:,i]))
        m[3,i] = gamma(4,np.linalg.norm(W_gt[:,i])) + 3*gamma(0,np.linalg.norm(W_gt[:,i])) - 6*gamma(2,np.linalg.norm(W_gt[:,i]))
    return [W_gt, v_gt, m]

def generateWeights_topk(d = 10, k = 5, s_gt = 0.75, kappa = 2 ):

    num_sparse = int(d*k*(1-s_gt))

    gauss_mat_u = np.random.normal(0.0, 1.0 , (d,k))
    gauss_mat_v = np.random.normal(0.0, 1.0 , (k,k))

    U, temp = np.linalg.qr(gauss_mat_u)
    V, temp = np.linalg.qr(gauss_mat_v)

    # U, V = np.linalg.qr(gauss_mat_u)
    diag = []
    v_gt = []
    v_choice = [1,-1]
    for iter in range(k):
        diag.append(1+1.*iter*(kappa-1)/(k-1))
        v_gt.append(random.choice(v_choice))

    Sigma = np.diag(diag)
    W_gt = np.dot(np.dot(U, Sigma), np.transpose(V))
    v_gt = np.asarray(v_gt)

    normW1 = []

    for i in range(d):

        for j in range(k):
            this_norm = np.linalg.norm(W_gt[i,j])
            normW1.append((this_norm, (i,j) ))

    normW1.sort()

    for i in range(num_sparse):
        d_ind = normW1[i][1][0]
        k_ind = normW1[i][1][1]

        W_gt[d_ind,k_ind] = 0.0


    m = np.zeros((4,k))
    for i in range(k):
        m[0,i] = gamma(1,np.linalg.norm(W_gt[:,i]))
        m[1,i] = gamma(2,np.linalg.norm(W_gt[:,i])) - gamma(0,np.linalg.norm(W_gt[:,i]))
        m[2,i] = gamma(3,np.linalg.norm(W_gt[:,i])) - 3*gamma(1,np.linalg.norm(W_gt[:,i]))
        m[3,i] = gamma(4,np.linalg.norm(W_gt[:,i])) + 3*gamma(0,np.linalg.norm(W_gt[:,i])) - 6*gamma(2,np.linalg.norm(W_gt[:,i]))
    return [W_gt, v_gt, m]


def generateData(W_gt, v_gt,n = 2000, test_n = 1000, d = 10):

    train_x = []
    train_y = []
    for iter in range(n):
        train_x.append(np.random.normal(0.0,1.,d))
        train_y.append(np.dot(phi(np.dot(train_x[iter], W_gt)),v_gt))
    train_x = np.asarray(train_x)
    train_y = np.transpose(np.asarray(train_y))

    test_x = []
    test_y = []
    for iter in range(test_n):
        test_x.append(np.random.normal(0.0,1.,d))
        test_y.append(np.dot(phi(np.dot(test_x[iter], W_gt)),v_gt))
    test_x = np.asarray(test_x)
    test_y = np.transpose(np.asarray(test_y))

    return [train_x, train_y, test_x, test_y]

