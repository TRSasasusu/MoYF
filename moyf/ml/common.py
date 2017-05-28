# coding: utf-8

import numpy as np

def mse(output_train, last_outputs):
    return 1 / (2 * len(output_train)) * (np.linalg.norm(output_train - last_outputs, axis=1) ** 2).sum()

def cross_entropy(output_train, last_outputs):
    return -(output_train * np.log(last_outputs)).sum()

def relu(x):
    return np.array([0 if value <= 0 else value for value in x])

def heaviside_step(x):
    return np.array([0 if value <= 0 else 1 for value in x])

def linear(x):
    return x

def constant_one(x):
    return np.ones(shape=(1, len(x)))

def softmax(x):
    total = np.exp(x).sum()
    return x / total

def dif(func, x):
    if func == relu:
        return heaviside_step(x)
    if func == linear:
        return constant_one(x)
    print('The differentiation of the function is not implemented.')
    return x;
