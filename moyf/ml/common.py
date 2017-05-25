# coding: utf-8

def mse():
    pass

def relu(x):
    return np.array([0 if value <= 0 else value for value in x])

def heaviside_step(x):
    return np.array([0 if value <= 0 else 1 for value in x])

def linear(x):
    return x

def constant_one(x):
    return np.ones(shape=(1, len(x)))

def dif(func, x):
    if func == relu:
        return heaviside_step(x)
    if func == linear:
        return constant_one(x)
