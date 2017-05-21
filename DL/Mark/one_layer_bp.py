#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import sys

w1 = 0
w2 = 0
b = 0
eta = 0.001
loss = 0
acc = 0

np.random.seed(3333)
if len(sys.argv) > 1:
    n_iter = int(sys.argv[1])
else:
    n_iter = 1000000
for i in range(n_iter):
    x1 = np.random.rand()
    x2 = np.random.rand()
    z_ = float(x1+x2 > 1)
    y = w1*x1 + w2*x2 + b
    z = 1.0 / (1+np.exp(-y))
    loss += -z_*np.log(z) - (1-z_)*np.log(1-z)
    w1 -= eta*(z-z_)*x1
    w2 -= eta*(z-z_)*x2
    b -= eta*(z-z_)
    acc += float(z_ == float(z > 0.5))

    if i%(n_iter/10) == 0:
        print("{}, loss {}, acc {}%".format(i, loss, 100*acc/(n_iter/10)))
        loss = 0
        acc = 0
