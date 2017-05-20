# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

__all__ = ["RNN"]

class RNN(object):
    """
    Simple BPTT RNN
    """

    def __init__(self, hidden_size, input_size):
        """
        `hidden_size` <int>: the size of hidden state cell
        `input_size` <int>: the shape of the input

        RNN:
            y_t = softmax(V.dot(s_t))
            s_t = tanh(W.dot(s_{t-1}) + U.dot(x_t))
        """
        self._hidden_size = hidden_size
        self._input_size = input_size
        self.W = np.zeros((hidden_size, hidden_size))
        self.U = np.zeros((hidden_size, input_size))
        self.V = np.zeros((input_size, hidden_size))

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def input_size(self):
        return self._input_size

    def forward_prop(self, x):
        """
        x: numpy array with shape (T, self.input_size)
            where T is the batch size
        """
        T = x.shape[0]
        states = np.zeros((T+1, self._hidden_size))
        outputs = np.zeros((T, self._input_size))
        for t in xrange(T):
            state = np.tanh(self.W.dot(state[t-1])+self.U.dot(x[t]))
            states[t] = state
            outputs[t] = self._softmax(self.V.dot(state))
        return (outputs, states)

    def predict(self, x):
        """
        x: numpy array with shape (T, self.input_size)
        """
        outputs, _ = self.forward_prop(x)
        return np.argmax(outputs, axis=1)

    def back_prop(self, x, y):
        """
        x: numpy array with shape (T, self.input_size), the input data
        y: numpy array with shape (T, self.input_size), the output
        """
        T = x.shape[0]
        outputs, states = self.forward_prop(x)
        dV = np.zeros(self.V.shape)
        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        delta = outputs
        delta[np.arange(T), y] -= 1
        for t in xrange(T):
            dV += np.outer(delta[t], states[t])
            delta_t = self.V.T.dot(delta[t]) * (1 - (states[t] ** 2))
            for b_step in np.arange(0, t+1)[::-1]:
                # b_step from t to 0
                dW += np.outer(delta_t, states[b_step-1])
                dU += np.outer(delta_t, x[b_step])
                delta_t = self.W.T.dot(delta_t)*(1-states[b_step-1]**2)
        dV /= T
        dU /= T
        dW /= T
        return [dV, dU, dW]

    def train(self, x, y, learn_rate=0.05):
        dV, dU, dW = self.back_prop(x, y)
        self.V -= learn_rate*dV
        self.U -= learn_rate*dU
        self.W -= learn_rate*dW

    def _softmax(self, x):
        """
        private softmax function
        softmax(x) = exp(x)/exp(x).sum()
        """
        e = np.exp(x)
        return e/e.sum()
