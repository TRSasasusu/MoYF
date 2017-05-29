# coding: utf-8

import numpy as np

def mse(output_train, last_outputs):
    return 1 / (2 * len(output_train)) * (np.linalg.norm(output_train - last_outputs, axis=1) ** 2).sum()

def cross_entropy(output_train, last_outputs):
    return -(output_train * np.log(last_outputs)).sum()

def relu(x):
    return np.maximum(x, 0)

def heaviside_step(x):
    return (np.sign(x) + 1) * 0.5

def linear(x):
    return x

def constant_one(x):
    return np.ones_like(x)

def softmax(x):
    return x / x.sum(axis=1)

def dif(func, x):
    if func == relu:
        return heaviside_step(x)
    if func == linear:
        return constant_one(x)
    print('The differentiation of the function is not implemented.')
    return x;
