#!/usr/bin/env python3
from __future__ import print_function
import numpy as np

class SimpleBackPropNetwork(object):

    def __init__(self, shapes):

        self._weights = []
        self._bias = []
        self._shapes = shapes

        for from_layer, to_layer in zip(shapes[0:-1], shapes[1:]):
            weight = np.random.randn(to_layer, from_layer)
            bias = np.random.randn(to_layer)
            self._weights.append(weight)
            self._bias.append(bias)

    def forward(self, x):
        """
        params
        ======
          - x <ndarray>: k x n array, where k is the input size, n is the number of samples.
        """
        if x.ndim == 1:
            x = x[:,None] # reshape 1D array into Nx1 matrix
        # activation from layer 0 to L (0 is input layer, L is the output layer)
        acts = [x.copy()]
        act = x
        for weight, bias in zip(self._weights, self._bias):
            act = weight.dot(act) + bias[:,None]
            acts.append(act)
            act = self._sigmoid(act)
        return acts

    def predict(self, x):
        acts = self.forward(x)
        y_hat = self._sigmoid(acts[-1])
        if x.ndim == 1:
            return y_hat.flatten()
        return y_hat

    def back_prop(self, x, y, learn_rate = 0.001):
        """
        params
        ======
          - x: k x N
          - y: m x N
        """
        acts = self.forward(x)
        L = len(self._weights)
        y_hat = self.predict(x)
        N = x.shape[1]

        delta = (y - y_hat)*y_hat*(1-y_hat) # m x N
        for i in range(1, L+1):
            act = acts[L-i]
            dW = delta.dot(self._sigmoid(act.T))/N
            db = delta.mean(axis=1)
            self._weights[-i] += learn_rate*dW
            self._bias[-i] += learn_rate*db
            weight = self._weights[-i]
            delta = weight.T.dot(delta)*act*(1-act)

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __str__(self):
        return "SimpleBackPropNetwork: {}".format("x".join(map(str, self._shapes)))

if __name__ == "__main__":

    np.random.seed(10)
    nn = SimpleBackPropNetwork([2, 10, 1])
    x = np.array([[0, 1, 1, 0],
                  [0, 0, 1, 1]])
    y = np.array([[1, 0, 1, 0]])

    for i in range(1, 10001):
        nn.back_prop(x, y)
        if i % 100 == 0:
            y_hat = nn.predict(x)
            loss = np.sqrt(((y - y_hat)**2).sum(axis=0)).mean()
            print("loss at iteration {}: {}".format(i, loss))

