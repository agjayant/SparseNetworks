
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import random, math
import matlab
import matlab.engine as me
from dataSetup import generateData, generateWeights
from recovAnalysis import recovery, structDiff
from trainerUtil import tensorInit, train


# In[ ]:


eng = me.start_matlab()


# In[ ]:


num_epoch = 25
epsilon = 1e-4 
recovery_delta = 1e-2
batch_size = 20
lr = 1e-3
test_n = 1000

## Baseline Values
d = 20
k = 5
thresh_gt = 0.20
n = 8000
noise_sd = 0
thresh_train = 0.15


# In[ ]:


## Vary n
exp_n = [2000, 4000, 6000, 8000, 10000]
w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)

logFile = open('log_n.txt','w')
for n in exp_n:
    
    train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
    train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

    print "Tensor Initialization for n = ", n
    tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)
    
    w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, 0 , num_epoch, 
                                         batch_size, lr, epsilon) 

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    recoveryVal_o = recovery(w_gt, v_gt, w_res_o, v_gt)
    print recoveryVal, recoveryVal_o

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    recoveryStructure_o = structDiff(w_gt, w_res_o, recovery_delta)
    print recoveryStructure, recoveryStructure_o
    
    logFile.write(str(n)+' '+str(recoveryVal)+' '+
                  str(recoveryVal_o)+' '+str(recoveryStructure)+' '+str(recoveryStructure_o)+'\n')

logFile.close()
n = 8000


# In[ ]:


# Vary d
exp_d = [10, 20, 40, 70, 100]

logFile = open('log_d.txt','w')
for d in exp_d:
    w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)
    train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
    train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

    print "Tensor Initialization for d = ", d
    tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)   

    w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, 0 , num_epoch, 
                                         batch_size, lr, epsilon) 

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    recoveryVal_o = recovery(w_gt, v_gt, w_res_o, v_gt)
    print recoveryVal, recoveryVal_o

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    recoveryStructure_o = structDiff(w_gt, w_res_o, recovery_delta)
    print recoveryStructure, recoveryStructure_o
    
    logFile.write(str(d)+' '+str(recoveryVal)+' '+
                  str(recoveryVal_o)+' '+str(recoveryStructure)+' '+str(recoveryStructure_o)+'\n')

logFile.close()
d = 20


# In[ ]:


# Vary d
exp_k = [2, 5, 10]

logFile = open('log_k.txt','w')
for k in exp_k:
    w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)
    train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
    train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

    print "Tensor Initialization for k = ", k
    tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)   

    w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, 0 , num_epoch, 
                                         batch_size, lr, epsilon) 

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    recoveryVal_o = recovery(w_gt, v_gt, w_res_o, v_gt)
    print recoveryVal, recoveryVal_o

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    recoveryStructure_o = structDiff(w_gt, w_res_o, recovery_delta)
    print recoveryStructure, recoveryStructure_o
    
    logFile.write(str(k)+' '+str(recoveryVal)+' '+
                  str(recoveryVal_o)+' '+str(recoveryStructure)+' '+str(recoveryStructure_o)+'\n')

logFile.close()
k = 5


# In[ ]:


# Vary thresh_gt
exp_thresh_gt = [0.05, 0.10, 0.15, 0.20, 0.25]

logFile = open('log_thresh_gt.txt','w')
for thresh_gt in exp_thresh_gt:
    w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)
    train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
    train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

    print "Tensor Initialization for thresh_gt = ", thresh_gt
    tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)   

    w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, 0 , num_epoch, 
                                         batch_size, lr, epsilon) 

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    recoveryVal_o = recovery(w_gt, v_gt, w_res_o, v_gt)
    print recoveryVal, recoveryVal_o

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    recoveryStructure_o = structDiff(w_gt, w_res_o, recovery_delta)
    print recoveryStructure, recoveryStructure_o
    
    logFile.write(str(thresh_gt)+' '+str(recoveryVal)+' '+
                  str(recoveryVal_o)+' '+str(recoveryStructure)+' '+str(recoveryStructure_o)+'\n')

logFile.close()
thresh_gt = 0.15


# In[ ]:


# Vary thresh_train
exp_thresh_train = [0, 0.05, 0.10, 0.15, 0.20, 0.25]

w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)
train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)
train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

print "Tensor Initialization for thresh_train"
tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

logFile = open('log_thresh_train.txt','w')
for thresh_train in exp_thresh_train:

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)   

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    print recoveryVal

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    print recoveryStructure
    
    logFile.write(str(thresh_train)+' '+str(recoveryVal)+' '+str(recoveryStructure)+'\n')

logFile.close()
thresh_train = 0.15


# In[ ]:


# Vary noise_sd
exp_noise_sd = [0.0, 0.1, 0.2, 0.3, 0.4]

w_gt, v_gt, m =  generateWeights(d, k, thresh_gt)
train_x, train_y, test_x, test_y = generateData(w_gt, v_gt, n, test_n, d)

logFile = open('log_noise_sd.txt','w')
for noise_sd in exp_noise_sd:

    train_y_noisy = train_y + np.random.normal(0, noise_sd, n)

    print "Tensor Initialization for noise_sd = ", noise_sd
    tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)   

    w_res_o, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, 0 , num_epoch, 
                                         batch_size, lr, epsilon) 

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    recoveryVal_o = recovery(w_gt, v_gt, w_res_o, v_gt)
    print recoveryVal, recoveryVal_o

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    recoveryStructure_o = structDiff(w_gt, w_res_o, recovery_delta)
    print recoveryStructure, recoveryStructure_o
    
    logFile.write(str(noise_sd)+' '+str(recoveryVal)+' '+
                  str(recoveryVal_o)+' '+str(recoveryStructure)+' '+str(recoveryStructure_o)+'\n')

logFile.close()
noise_sd = 0


# In[ ]:


eng.quit()

