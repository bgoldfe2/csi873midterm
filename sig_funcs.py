# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:17:17 2017

@author: Bruce
"""
"""
sig_funcs.py - Utility module for listing various sigmoid functions

"""
import numpy as np

# The sigmoid function and its derivative
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# The tangent function and its derivative
def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - tanh(z) * tanh(z)