#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np

def _sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1.0/(1+np.exp(-x))

class SimpleBackPropNetwork(object):
    """
    Simple Back Propagation Network

    loss function: quadratic
    activation: sigmoid
    """
    def __init__(self, shapes):
        self._weights = []
        self._bias = []
        for input_shape, output_shape in zip(shapes[:-1], shapes[1:]):
            weight = 0.5 - np.random.randn(input_shape, output_shape)
            bias = 0.5 - np.random.randn(output_shape)
            self._weights.append(weight)
            self._bias.append(bias)
        self._shapes = shapes

    def forward(self, x):
        """
        params
        ======
        x: Nxk array. k is the input size

        return acts, zs
        """
        acts = [x.copy()] # activations
        act = acts[0]
        zs = [] # z vectors
        for w, b in zip(self._weights, self._bias):
            z = np.dot(act, w) + b
            act = _sigmoid(z)
            acts.append(act)
            zs.append(z)
        return acts, zs

    def predict(self, x):
        act = self.forward(x)[0][-1]
        return act

    def back_prop(self, x, y, eta=0.01):
        N = x.shape[0]
        acts, zs = self.forward(x)
        y_hat = acts[-1]
        delta = (y_hat - y)*_sigmoid(zs[-1], deriv=True)
        num_layers = len(acts)
        dbs = [None for _ in range(num_layers-1)]
        dbs[-1] = delta.mean(axis=0)
        dWs = [None for _ in range(num_layers-1)]
        dW = np.dot(acts[-2].T, delta)/N
        dWs[-1] = dW
        for l in range(2, num_layers):
            z = zs[-l]
            sig_deriv = _sigmoid(z, deriv=True)
            delta = np.dot(delta, self._weights[-l+1].T)*sig_deriv
            dbs[-l] = delta.mean(axis=0)
            dWs[-l] = np.dot(acts[-l-1].T, delta)/N

        for i, (dW, db) in enumerate(zip(dWs, dbs)):
            self._weights[i] -= eta*dW
            self._bias[i] -= eta*db

    def __repr__(self):
        return "{}: {}".format(self.__class__, "x".join(map(str, self._shapes)))

if __name__ == "__main__":
    x = np.array([[0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 1]])
    y = np.array([0, 0, 1, 0]).reshape((4, 1))
    nn = SimpleBackPropNetwork([2, 2, 1])
    for i in range(700):
        nn.back_prop(x, y, 0.003)

    print(nn.predict(np.array([0, 0])))
    print(nn.predict(np.array([1, 0])))
    print(nn.predict(np.array([0, 1])))