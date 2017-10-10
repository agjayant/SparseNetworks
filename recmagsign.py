import numpy as np
import random, math
from myTensorUtil import *
from powMeth import *

def recmagsgn(V, U, X, y, m , W_gt):

    ### l1 and l2 :: TODO
    l1 = 1
    l2 = 2

    d = np.shape(X)[1]
    k = np.shape(U)[0]
    divInd = int(len(X)/2)

    # Partition
    X1 = X[:divInd]
    y1 = y[:divInd]

    X2 = X[divInd:]
    y2 = y[divInd:]

    alpha = np.random.normal(0.0, 0.1 , d)
    alpha = alpha/np.linalg.norm(alpha)

    ## TODO: Assuming l1=1 and l2=2
    Q1 = getM1(X1, y1)
    Q2 = multLnr1(getM2(X2,y2), V).flatten()

    Vu = []
    UU = []

    for ind in range(len(U)):
        Vu.append(np.dot(V, U[ind]))
        UU.append(np.dot( np.transpose([U[ind]]), [U[ind]] ).flatten())

    ## Estimating z
    z_old = np.zeros(k)
    z_new = np.zeros(k)

    T = 10000
    for iterT in range(T):
        for ind in range(k):

            mult_fact = np.zeros(d)
            for j in range(k):
                if j != ind:
                    mult_fact += z_old[j]*Vu[j]
            div_fact = np.dot(np.transpose(Vu[ind]), Vu[ind])
            z_new[ind] = (np.dot(np.transpose(Q1), Vu[ind]) + np.dot(np.transpose(Vu[ind]), Q1) -
                                 np.dot(np.transpose(Vu[ind]), mult_fact) -
                                  np.dot(np.transpose(mult_fact), Vu[ind]) )/(2*div_fact)
        z_old = z_new

    ## Estimating r
    r = np.dot(np.linalg.inv(np.dot(UU, np.transpose(UU))),
                                                   np.dot(UU,np.transpose([Q2]) ) )
    v = np.sign(r*np.transpose([m[l2-1]]))
    s = np.sign(v*np.transpose([z_new])*np.transpose([m[l2-1]]))

    p = 1 ## p + 1 is degree of homogenity

    w = []
    for ind in range(k):

        w.append(s[ind]*np.math.pow(abs(z_new[ind]*np.linalg.norm(W_gt[:,ind])**(p+1)/(m[l1-1,ind]*np.math.pow( np.dot( [alpha],
                    np.transpose([Vu[ind]]) ) ,l1-1))), 1.0/(p+1))*Vu[ind])
    w = np.asarray(w)

    return w,v

