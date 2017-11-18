import numpy as np
import tensorflow as tf
import random, math
from myTensorUtil import *
from powMeth import *
from recmagsign import *
import matlab

def tensorInit(X, y, W_gt, m, k, eng):
    divInd = int(len(X)/3)

    # Partition
    X1 = X[:divInd]
    y1 = y[:divInd]

    X2 = X[divInd:2*divInd]
    y2 = y[divInd:2*divInd]

    X3 = X[2*divInd:]
    y3 = y[2*divInd:]

    ## P2
    ## Estimating P2 as M2

    ## Power Method
    V = powMeth(k, X1, y1)

    ## R3
    R3 = multLnr2(getM3(X2, y2), V)

    #### KCL
    R = matlab.double(R3.flatten().tolist())
    U = eng.notf_frompy(R, 100, k)
    U = np.asarray(U)
    #U = R3[0]

    ## RecMagSign
    return recmagsgn(V, U, X3, y3, m, W_gt)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=1.0)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    """
    h    = tf.square(tf.nn.relu((tf.matmul(X, w_1))))
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def train(train_x, train_y, test_x, test_y, tensorWeights, v_gt, s=0.15, T = 25, batch_size = 10, lr = 1e-3,epsilon = 1e-4, T_pretrain = 5, random=False):

    d = np.shape(train_x)[1]
    n = np.shape(train_x)[0]
    test_n = np.shape(test_x)[0]
    k = np.shape(v_gt)[0]

    X = tf.placeholder("float64", shape=[None, d])
    y = tf.placeholder("float64", shape=[None, 1])

    # Weight initializations
    if random:
        w_1 = init_weights((d, k))
        w_1 = tf.Variable(tf.cast(w_1, tf.float64))
    else:
        w_1 = tf.Variable(np.transpose(tensorWeights[0]))

    # w_2 = []
    # v_choice = [1,-1]
    # for iter in range(k):
    #     w_2.append(random.choice(v_choice))
    # w_2 = tf.cast(tf.Variable(np.transpose(np.asarray([w_2])), trainable=False), tf.float32)
    # w_2 = init_weights((k, 1))

    # w_2 = tf.Variable(tensorWeights[1], trainable=False)

    w_2 = tf.Variable( np.transpose([v_gt]), trainable=False)
    w_2 = tf.cast(w_2, tf.float64)

    # Forward propagation
    yhat  = forwardprop(X, w_1, w_2)

    # Backward propagation
    cost = tf.losses.mean_squared_error(y, yhat)
    updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # Run SGD
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    train_loss = []
    test_loss = []

    for epoch in range(T):

        ## Shuffling
        permList = np.random.permutation(range(n))
        train_x = train_x[permList]
        train_y = train_y[permList]

        ## Iterative Hard Thresholding
        ## after every epoch

        if epoch > T_pretrain:

            cur_w1 =  w_1.eval(session=sess)
            for i in range(d):
                for j in range(k):
                    this_norm = np.linalg.norm(cur_w1[i,j])
                    if this_norm <= s:
                        cur_w1[i,j] = 0.0
            assign_op = tf.assign(w_1, cur_w1)
            sess.run(assign_op)

        i = 0
        for iter in range(int(n/batch_size)):
            sess.run(updates, feed_dict={X: train_x[i: i + batch_size], y: train_y[i: i + batch_size].reshape(batch_size,1)})
            i = (i + batch_size)%n
    #     train_accuracy = np.mean((train_y - sess.run(yhat, feed_dict={X:train_x})) <= epsilon)
        loss = sess.run(cost, feed_dict={X:train_x, y:train_y.reshape(n,1)})
        testLoss = sess.run(cost, feed_dict={X:test_x, y:test_y.reshape(test_n,1)})

        print "Epoch = ", epoch+1,"training loss: ", loss ," test loss: ",testLoss

        train_loss.append(loss)
        test_loss.append(testLoss)

    final_w1 = sess.run(w_1)
    sess.close()
    return final_w1, train_loss, test_loss

def train_topk(train_x, train_y, test_x, test_y, tensorWeights, v_gt, s=0.75, T = 25, batch_size = 10, lr = 1e-3,epsilon = 1e-4, T_pretrain = 5, random=False):

    d = np.shape(train_x)[1]
    n = np.shape(train_x)[0]
    test_n = np.shape(test_x)[0]
    k = np.shape(v_gt)[0]

    num_sparse = int( d*k*( 1 - s ) )

    X = tf.placeholder("float64", shape=[None, d])
    y = tf.placeholder("float64", shape=[None, 1])

    # Weight initializations
    if random:
        w_1 = init_weights((d, k))
        w_1 = tf.Variable(tf.cast(w_1, tf.float64))
    else:
        w_1 = tf.Variable(np.transpose(tensorWeights[0]))

    # w_2 = []
    # v_choice = [1,-1]
    # for iter in range(k):
    #     w_2.append(random.choice(v_choice))
    # w_2 = tf.cast(tf.Variable(np.transpose(np.asarray([w_2])), trainable=False), tf.float32)
    # w_2 = init_weights((k, 1))

    # w_2 = tf.Variable(tensorWeights[1], trainable=False)

    w_2 = tf.Variable( np.transpose([v_gt]), trainable=False)
    w_2 = tf.cast(w_2, tf.float64)

    # Forward propagation
    yhat  = forwardprop(X, w_1, w_2)

    # Backward propagation
    cost = tf.losses.mean_squared_error(y, yhat)
    updates = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    # Run SGD
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    train_loss = []
    test_loss = []

    for epoch in range(T):

        ## Shuffling
        permList = np.random.permutation(range(n))
        train_x = train_x[permList]
        train_y = train_y[permList]

        ## Iterative Hard Thresholding
        ## after every epoch

        if epoch > T_pretrain:

            cur_w1 =  w_1.eval(session=sess)
            normW1 = []
            for i in range(d):
                for j in range(k):
                    this_norm = np.linalg.norm(cur_w1[i,j])
                    normW1.append( (this_norm, (i,j) ) )

            normW1.sort()

            for i in range(num_sparse):
                ind_d = normW1[i][1][0]
                ind_k = normW1[i][1][1]

                cur_w1[ind_d, ind_k] = 0.0

            assign_op = tf.assign(w_1, cur_w1)
            sess.run(assign_op)

        i = 0
        for iter in range(int(n/batch_size)):
            sess.run(updates, feed_dict={X: train_x[i: i + batch_size], y: train_y[i: i + batch_size].reshape(batch_size,1)})
            i = (i + batch_size)%n
    #     train_accuracy = np.mean((train_y - sess.run(yhat, feed_dict={X:train_x})) <= epsilon)
        loss = sess.run(cost, feed_dict={X:train_x, y:train_y.reshape(n,1)})
        testLoss = sess.run(cost, feed_dict={X:test_x, y:test_y.reshape(test_n,1)})

        print "Epoch = ", epoch+1,"training loss: ", loss ," test loss: ",testLoss

        train_loss.append(loss)
        test_loss.append(testLoss)

    final_w1 = sess.run(w_1)
    sess.close()
    return final_w1, train_loss, test_loss
