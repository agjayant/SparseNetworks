import numpy as np
import tensorflow as tf
import random, math
import matlab
import matlab.engine as me
from dataSetup import generateData, generateWeights, generateWeights_topk
from recovAnalysis import recovery, structDiff
from trainerUtil import tensorInit, train, train_topk, init_weights
import argparse

argparser  = argparse.ArgumentParser(description="Experiments n")
argparser.add_argument('-w', '--winit', help='Weight Initialization', default='tensor')
argparser.add_argument('-i', '--iht', help='IHT Algorithm Type', default='topk')
argparser.add_argument('-l', '--log', help='Log File Name', default='log_k.txt')
argparser.add_argument('-n', '--numT', help='Number of Trials', default=5)
argparser.add_argument('-gt', '--gtinit', help='Type of Ground Truth Weight Init', default=False)
args = argparser.parse_args()


num_epoch = 25
num_epoch_pretrain = 5
epsilon = 1e-4
recovery_delta = 1e-2
batch_size = 20
lr = 1e-3
test_n = 1000

if args.winit == 'random':
    random = True
else:
    random = False

iht = args.iht

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

num_trials = args.numT
## Vary sparse_gt
exp_noise_sd = [0.0, 0.1, 0.2, 0.3, 0.4]

logFile = open(args.log,'w')

for noise_sd in exp_noise_sd:

    recovery_this = []
    recovery_o_this = []

    truep = []
    truep_o = []
    truen = []
    truen_o = []

    for trial in range(num_trials):

        print "Experiment Starting for noise_sd = ",noise_sd, " trial: ", trial

        if iht == 'topk':
            w_gt, v_gt, m =  generateWeights_topk(d, k, sparse_gt, bool(args.gtinit))
        else:
            w_gt, v_gt, m =  generateWeights(d, k, thresh_gt, bool(args.gtinit))

        train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
        train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

        if random:
            tensorWeights = []
        else:
            tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)


        if iht == 'topk':
            w_res, train_loss, test_loss = train_topk(train_x, train_y_noisy, test_x,
                                                 test_y,
                                                 tensorWeights,v_gt, sparse_train,
                                                 num_epoch, batch_size, lr, epsilon,
                                                 num_epoch_pretrain, random)

            w_res_o, train_loss, test_loss = train_topk(train_x, train_y_noisy, test_x,
                                                 test_y,
                                                 tensorWeights,v_gt, 1.0 , num_epoch,
                                                 batch_size, lr, epsilon,
                                                 num_epoch_pretrain, random)


        else:
            w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y,
                                                 tensorWeights,v_gt, thresh_train,
                                                 num_epoch, batch_size, lr, epsilon,
                                                 num_epoch_pretrain, random)

            w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x,
                                                 test_y,
                                                 tensorWeights,v_gt, 0.0 , num_epoch,
                                                 batch_size, lr, epsilon,
                                                 num_epoch_pretrain, random)

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

    avg_recov = np.mean(recovery_this)
    std_recov = np.std(recovery_this)

    avg_recov_o = np.mean(recovery_o_this)
    std_recov_o = np.std(recovery_o_this)

    avg_truen = np.mean(truen)
    avg_truep = np.mean(truep)
    avg_truen_o = np.mean(truen_o)
    avg_truep_o = np.mean(truep_o)

    std_truen = np.std(truen)
    std_truep = np.std(truep)
    std_truen_o = np.std(truen_o)
    std_truep_o = np.std(truep_o)


    logFile.write(str(noise_sd)+' '+str(avg_recov)+' '+str(std_recov) + ' '+str(avg_recov_o)+' '+str(std_recov_o)+' ' )
    logFile.write(str(avg_truen) + ' '+ str(std_truen)+' ')
    logFile.write(str(avg_truen_o) + ' '+ str(std_truen_o)+' ')
    logFile.write(str(avg_truep) + ' '+ str(std_truep)+' ')
    logFile.write(str(avg_truep_o) + ' '+ str(std_truep_o))
    logFile.write('\n')

logFile.close()

if not random:
    eng.quit()

