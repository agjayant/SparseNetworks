import numpy as np
import tensorflow as tf
import random, math
import matlab
import matlab.engine as me
from dataSetup import generateData, generateWeights, generateWeights_topk
from recovAnalysis import recovery, structDiff
from trainerUtil import tensorInit, train, train_topk, init_weights

num_epoch = 25
num_epoch_pretrain = 5
epsilon = 1e-4
recovery_delta = 1e-2
batch_size = 20
lr = 1e-3
test_n = 1000
random= True

if not random:
    eng = me.start_matlab()
## Baseline Values
d = 20
k = 5
thresh_gt = 0.15
sparse_gt = 0.75
n = 6000
noise_sd = 0
thresh_train = 0.15
sparse_train = 0.75

num_trials = 5

## Vary n
exp_n = [2000, 4000, 6000, 8000, 10000]

logFile = open('log_n.txt','w')
for n in exp_n:

    recovery_this = []
    recovery_o_this = []

    truep = []
    truep_o = []
    truen = []
    truen_o = []

    for trial in range(num_trials):

        w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)

        train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
        train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

        print "Weight Initialization for n = ", n

        if random:
            tensorWeights = []
        else:
            tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)


        w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y,
                                             tensorWeights,v_gt, thresh_train, num_epoch,
                                             batch_size, lr, epsilon, num_epoch_pretrain, random)

        w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y,
                                             tensorWeights,v_gt, 0 , num_epoch,
                                             batch_size, lr, epsilon, num_epoch_pretrain, random)

        recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
        recoveryVal_o = recovery(w_gt, v_gt, w_res_o, v_gt)

        recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
        recoveryStructure_o = structDiff(w_gt, w_res_o, recovery_delta)

        recovery_this.append(recoveryVal)
        recovery_o_this.append(recoveryVal_o)

        truen.append(recoveryStructure[2])
        truen_o.append(recoveryStructure_o[2])
        truep.append(recoveryStructure[3])
        truep_o.append(recoveryStructure_o[3])

    # print recoveryStructure, recoveryStructure_o
    # print recoveryVal, recoveryVal_o

    avg_recov = np.mean(recovery_this)
    std_recov = np.std(recovery_this)

    avg_recov_o = np.mean(recovery_o_this)
    std_recov_o = np.std(recovery_o_this)

    avg_truen = np.mean(truen)
    avg_truep = np.mean(truep)
    avg_truen_o = np.mean(truen_o)
    avg_truep_o = np.mean(truep_o)

    std_truen = np.mean(truen)
    std_truep = np.mean(truep)
    std_truen_o = np.mean(truen_o)
    std_truepn_o = np.mean(truep_o)


    # print "Average Recovery for n= ", n, "  ", avg_recov
    # print "Standard Deviation of Recovery for n= ", n, "  ", std_recov

    logFile.write(str(n)+' '+str(avg_recov)+' '+str(std_recov) + ' '+str(avg_recov_o)+' '+str(std_recov_o)+' ' )
    logFile.write(str(avg_truen) + ' '+ str(std_truen)+' ')
    logFile.write(str(avg_truen_o) + ' '+ str(std_truen_o)+' ')
    logFile.write(str(avg_truep) + ' '+ str(std_truep)+' ')
    logFile.write(str(avg_truep_o) + ' '+ str(std_truep_o))
    logFile.write('\n')
    # logFile.write(str(n)+' '+str(recoveryVal)+' '+
                  # str(recoveryVal_o)+' '+str(recoveryStructure)+' '+str(recoveryStructure_o)+'\n')

logFile.close()

if not random:
    eng.quit()

