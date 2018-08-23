#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np

def load_data_set():
    """
    加载数据集
    :return:
    """
    data_features = []
    data_labels = []
    with open('../../../../input/5.Logistic/TestSet.txt', 'r') as f:
        for line in f.readlines():
            line = line.split()
            #数据集的属性集，1.0表示初始值偏差为1.0
            data_features.append([1.0, np.float(line[0]), np.float(line[1])])
            data_labels.append(int(line[2]))
    data_features = np.array(data_features)
    data_labels = np.array(data_labels)
    return data_features, data_labels

def sigmoid(z):
    """
    compute the sigmoid of x
    param:
    x -- a scalar or numpy array of any size
    return:
    s -- sigmoid(x)
    """
    s = 1.0 / (1 + 1 / np.exp(z))
    return s

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + dataMatrix.transpose() * error
    return np.array(weights)
def fordward_propagation(W,X,Y):
    """

    :param W: 权重系数
    :param X:
    :param Y:
    :return:
     cost -- 代价
    """
    m = X.shape[1]
    A = sigmoid(np.dot(X,W.T))
    cost = (1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    return cost
def back_propagation(W,X,Y):
    A = sigmoid(np.dot(X, W.T))
    m = X.shape[0]
    dW = (1.0/m) * np.dot( (A-Y).T, X)
    return dW
def optimize(W,X,Y,num_iters, learning_rate):
    for i in range(num_iters):
        cost = fordward_propagation(W,X,Y)
        dW = back_propagation(W,X,Y)

        W = W - learning_rate*dW

        if i % 100 == 0:
            print(cost)
    return W

def test():
    X,Y = load_data_set()
    Y.reshape(100,1)
    print(X.shape,Y.shape)
    n = X.shape[1]
    W = np.zeros(shape=(1,n))
    W = optimize(W,X,Y,20000,0.01)
    print(W)



if __name__ =="__main__":
    test()
