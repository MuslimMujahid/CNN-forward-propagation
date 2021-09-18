import numpy as np
from .layers import Layer
from tabulate import tabulate


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers
        self.has_run = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.has_run = True

        output = X
        for layer in self.layers:
            output = layer(output)

        return output

    def summary(self):

        if not self.has_run:
            print("Do predict first before get summary!")
            return

        table = []
        heads = ["Layer (type)", "Output Shape", "Params"]
        for layer in self.layers:
            table.append([f'{layer.name} ({layer.__name__()})',
                          layer.output_shape, layer.param])
        print(tabulate(table, headers=heads, tablefmt="github"))
