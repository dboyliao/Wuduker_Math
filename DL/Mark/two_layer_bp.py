#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import sys

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1.0-x)
    return 1.0/(1+np.exp(-x))

if len(sys.argv) > 1:
    n_iter = int(sys.argv[1])
else:
    n_iter = 10000

# layer 2 (output)
w41, w42, w43 = [0.5 - np.random.rand() for _ in range(3)]
b4 = 0.5 - np.random.rand()

# layer 1 
w11, w12, w21, w22, w31, w32 =  [0.5 - np.random.rand() for _ in range(6)]
b1, b2, b3 = [0.5 - np.random.rand() for _ in range(3)]

z1, z2, z3, z4, = 0, 0, 0, 0
eta = 0.01
loss = 0
acc = 0
for i in range(n_iter):
    x1 = np.random.rand()
    x2 = np.random.rand()
    z_ = float(x1 + x2 > 0.5)
    # forward
    y1 = w11*x1 + w12*x2 + b1
    z1 = sigmoid(y1)
    y2 = w21*x1 + w22*x2 + b2
    z2 = sigmoid(y2)
    y3 = w31*x1 + w32*x2 + b3
    z3 = sigmoid(y3)

    y4 = w41*z1 + w42*z2 + w43*z3 + b4
    z4 = sigmoid(y4)
    loss += -z_*np.log(z4)-(1-z_)*np.log(1-z4) 
    acc += float(z_ == float(z4 > 0.5))

    update_w41 = eta*(z4 - z_)*z1
    update_w42 = eta*(z4 - z_)*z2
    update_w43 = eta*(z4 - z_)*z3
    update_b4 = eta*(z4 - z_)

    update_w11 = eta*(z4 - z_)*w41*sigmoid(z1, deriv=True)*x1
    update_w12 = eta*(z4 - z_)*w41*sigmoid(z1, deriv=True)*x2
    update_b1 = eta*(z4 - z_)*w41*sigmoid(z1, deriv=True)
    update_w21 = eta*(z4 - z_)*w42*sigmoid(z2, deriv=True)*x1
    update_w22 = eta*(z4 - z_)*w42*sigmoid(z2, deriv=True)*x2
    update_b2 = eta*(z4 - z_)*w42*sigmoid(z2, deriv=True)
    update_w31 = eta*(z4 - z_)*w43*sigmoid(z3, deriv=True)*x1
    update_w32 = eta*(z4 - z_)*w43*sigmoid(z3, deriv=True)*x2
    update_b3 = eta*(z4 - z_)*w43*sigmoid(z3, deriv=True)

    w41 -= update_w41
    w42 -= update_w42
    w43 -= update_w43
    b4 -= update_b4
    w11 -= update_w11
    w12 -= update_w12
    b1 -= update_b1
    w21 -= update_w21
    w22 -= update_w22
    b2 -= update_b2
    w31 -= update_w31
    w32 -= update_w32
    b3 -= update_b3

    if i % (n_iter/10) == 0:
        print("{}, loss {}, acc {}".format(i, loss, 100*acc/(n_iter/10)))
        loss = 0
        acc = 0
