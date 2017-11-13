
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
n = 4000
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

    tensorWeights = tensorInit(train_x, train_y_noisy, w_gt, m ,k, eng)

    w_res, train_loss, test_loss = train(train_x, train_y_noisy, test_x, test_y, 
                                         tensorWeights,v_gt, thresh_train, num_epoch, 
                                         batch_size, lr, epsilon)   

    recoveryVal = recovery(w_gt, v_gt, w_res, v_gt)
    print recoveryVal

    recoveryStructure = structDiff(w_gt, w_res, recovery_delta)
    print recoveryStructure
    
    logFile.write(str(n)+' '+str(recoveryVal)+' '+str(recoveryStructure)+'\n')

logFile.close()
n = 4000


# In[ ]:


eng.quit()

