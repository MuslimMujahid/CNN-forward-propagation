import numpy as np

def linear(x):
    return x

def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    exp = np.exp(x)
    return exp / np.reshape(np.sum(exp, axis=1), (exp.shape[0], 1))
  
ACTIVATION_FUNCTIONS = {
  "linear": linear,
  "sigmoid": sigmoid,
  "relu": relu,
  "softmax": softmax
}