import numpy as np
import random, math
import itertools


def structDiff(w_gt, w_res, delta = 1e-4):
    '''
    Returns :
    False Alarm --  GT - 0 Res- Non 0
    Mis Detection -- GT - Non 0 Res - 0
    True Positives -- GT - 0 Res - 0
    True Negatives -- GT - Non 0 Res - Non 0
    '''

    w_gt = w_gt.flatten()
    w_res = w_res.flatten()

    num_w = len(w_gt)
    assert(num_w == len(w_res)), 'Dimension not Equal\n'

    false_alarm, mis_det, tr_positive, tr_negative = 0,0,0,0

    for w in range(num_w):
        if abs(w_gt[w]) < delta and abs(w_res[w]) > delta :
            false_alarm += 1
        elif abs(w_gt[w]) > delta and abs(w_res[w]) < delta :
            mis_det += 1
        elif abs(w_gt[w]) < delta and abs(w_res[w]) < delta :
            tr_positive += 1
        elif abs(w_gt[w]) > delta and abs(w_res[w]) > delta :
            tr_negative += 1

    return [false_alarm, mis_det, tr_positive, tr_negative]


def recovery(W_gt, v_gt, w_res, v_res ):

    k = np.shape(v_gt)[0]
    permList = list(itertools.permutations(range(k)))

    w_p = np.transpose(w_res)

    v_p = v_res

    min_v = 100
    for perm in permList:
        w_pi = w_p[list(perm)]
        v_pi = v_p[list(perm)]

        w_gt = np.transpose(W_gt)

        if sum(v_gt == v_pi) == k:
            max_diff = 0
            for i in range(k):
                diff = np.linalg.norm(w_pi[i]-w_gt[i])/np.linalg.norm(w_gt)
                if diff > max_diff:
                    max_diff = diff
            if max_diff < min_v:
                min_v = max_diff
        else:
            continue
    return min_v
