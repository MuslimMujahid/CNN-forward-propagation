import numpy as np


def linear(x):
    return x


def sigmoid(x):
    return 1/(1 + np.exp(-1*x))


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# def dsigmoid(x):
#     return sigmoid(x) * (1 - sigmoid(x))
def dsigmoid(x):
    return x * (1 - x)


def drelu(x):
    return (x > 0) * 1.0

def dlinear(x):
  return x

ACTIVATION_FUNCTIONS = {
    "linear": linear,
    "sigmoid": sigmoid,
    "relu": relu,
    "softmax": softmax,
    "dsigmoid": dsigmoid,
    "drelu": drelu,
    "dlinear": dlinear
}
