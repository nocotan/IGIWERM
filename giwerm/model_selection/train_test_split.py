#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Model Selection Module
Split arrays or matrices into random train and test subsets as follow:
Cortes, Corinna, Mehryar Mohri, Michael Riley, and Afshin Rostamizadeh. 2008.
“Sample Selection Bias Correction Theory.” In Algorithmic Learning Theory,
38–53. Springer Berlin Heidelberg.
"""

import numpy as np


def train_test_split(x, y, gamma=16):
    w = np.random.rand(*x[0].shape)
    sigma = np.std(x @ w.T)

    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(len(x)):
        v = gamma * w.T @ x[i]

        p = 1 / (1 + np.exp(v)) / sigma
        q = np.random.rand(1)
        if p < q:
            train_x.append(x[i])
            train_y.append(y[i])
        else:
            test_x.append(x[i])
            test_y.append(y[i])

    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x, test_y = np.array(test_x), np.array(test_y)
    return train_x, train_y, test_x, test_y
