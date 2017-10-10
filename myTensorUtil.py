import numpy as np
import random, math

def outer3(a,b,c):
    A = np.outer(a,b)
    B = []
    for third in c:
        B.append(A*third)
    return np.asarray(B)

def outer4(a,b,c,d):
    A = outer3(a,b,c)
    B = []
    for fourth in d:
        B.append(A*fourth)
    return np.asarray(B)

def outer3I(x):
    return outer3(x,x,x)

def outer4I(x):
    return outer4(x,x,x,x)

def specOuterI(x):
    d = len(x)
    iden = np.identity(d)
    final = np.zeros([d,d,d])
    for i in range(d):
        final += outer3(x, iden[i], iden[i]) + outer3(iden[i], x, iden[i])+ outer3(iden[i], iden[i],x)
    return final

def specOuterMat(M):
    d = np.shape(M)[0]
    ## TODO
    return np.zeros([d,d,d,d])

def multLnr(M, argList):
    if len(np.shape(M)) == 3:
        a,b,c = argList

        assert(np.shape(M)[0] == np.shape(a)[0])
        assert(np.shape(M)[1] == np.shape(b)[0])
        assert(np.shape(M)[2] == np.shape(c)[0])
        ## HardCoding Here :: TODO
        res = np.zeros([np.shape(a)[-1],np.shape(b)[-1] ])
        for itera in range(np.shape(a)[0]):
            for iterb in range(np.shape(b)[0]):
                for iterc in range(np.shape(c)[0]):
                    res += M[itera, iterb, iterc]*c[iterc]*np.outer(a[itera], b[iterb])
        return res
    else:
        assert(len(np.shape(M)) == 4)
        a,b,c,d = argList
        if len(np.shape(c)) == 2:
            pass
        else:
            res = np.zeros([np.shape(a)[-1],np.shape(b)[-1] ])
            for itera in range(np.shape(a)[0]):
                for iterb in range(np.shape(b)[0]):
                    for iterc in range(np.shape(c)[0]):
                        for iterd in range(np.shape(d)[0]):
                            res += M[itera, iterb, iterc]*c[iterc]*d[iterd]*np.outer(a[itera], b[iterb])
            return res

def multLnr1(M, V):
    d = np.shape(M)[0]
    k = np.shape(V)[1]

    res = np.zeros((k,k))

    for itera in range(d):
        for iterb in range(d):
            res += M[itera, iterb]*np.outer(V[itera], V[iterb])

    return res

def multLnr2(P3, V):
    d = np.shape(P3)[0]
    k = np.shape(V)[1]

    res = np.zeros((k,k,k))

    for itera in range(d):
        for iterb in range(d):
            for iterc in range(d):
                res += P3[itera, iterb, iterc]*outer3(V[itera], V[iterb], V[iterc])

    return res

