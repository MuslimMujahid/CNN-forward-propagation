import numpy as np
from .layers import Layer


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def predict(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer(output)

        return output
