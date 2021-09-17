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

class Detector(Layer):
    def __init__(self, units: int, activation: str, name: str) -> None:
        super().__init__(activation, name)
        self.units = units
        # self.W = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.detect(X)

    def detect(self, X: np.ndarray) -> np.ndarray:
        relu = ACTIVATION_FUNCTIONS["relu"]
        output = relu(X)
        return output

class Pooling(Layer):
    def __init__(self, units: int, activation: str, name: str, size=2, stride=2) -> None:
        super().__init__(activation, name)
        self.units = units
        self.size = size
        self.stride = stride

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.pool(X)

    def pool(self, X: np.ndarray) -> np.ndarray:
        output = np.zeros((np.uint16((X.shape[0]-self.size+1)/self.stride+1), np.uint16((X.shape[1]-self.size+1)/self.stride+1), X.shape[-1]))
        for map_num in range(X.shape[-1]):
            r2 = 0
            for r in np.arange(0,X.shape[0]-self.size+1, self.stride):
                c2 = 0
                for c in np.arange(0, X.shape[1]-self.size+1, self.stride):
                    output[r2, c2, map_num] = np.max([X[r:r+self.size,  c:c+self.size]])
                    c2 = c2 + 1
                r2 = r2 +1
        return output