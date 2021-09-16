import numpy as np
from .functions import ACTIVATION_FUNCTIONS


class Layer:
    def __init__(self, activation: str, name: str) -> None:
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.name = name


class Dense(Layer):
    def __init__(self, units: int, activation: str, name: str) -> None:
        super().__init__(activation, name)
        self.units = units
        self.W = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if (not self.W):
            self.W = np.random.rand(X.shape[1], self.units)

        net = np.dot(X, self.W)
        output = self.activation(net)
        return output
