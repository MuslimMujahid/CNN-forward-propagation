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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1, learning_rate=0.5) -> np.ndarray:
        layer_names = [l.__name__() for l in self.layers]
        flatten_idx = layer_names.index('Flatten')

        n_layers = len(self.layers)
        for _ in range(epochs):
            # Backpropagation for fully connected layer
            lst_dE_dw = []
            dE_dnet = None

            # Forward propagation
            yHat = self.predict(X)
            print(yHat)
            # Back propagation
            dE_dnet, dE_dw = self.layers[-1].backward(
                yHat, y)
            lst_dE_dw.insert(0, dE_dw)
            if (n_layers-flatten_idx > 1):
                for i in range(n_layers-2, flatten_idx, -1):
                    dE_dnet, dE_dw = self.layers[i].backward(
                        dE_dnet, next_layer=self.layers[i+1])
                    lst_dE_dw.insert(0, dE_dw)

            # Update weights
            for idx, dE_dw in enumerate(lst_dE_dw, flatten_idx + 1):
                self.layers[idx].W -= (learning_rate * dE_dw)

            # Backpropagation for convolutional layer
            for i in range(flatten_idx-1, -1, -1):
                backOut, weightOut = self.layers[i].backward(dE_dnet)

    def summary(self):
        table = []
        heads = ["Layer (type)", "Output Shape", "Params"]
        prev_output_shape = None
        total_param = 0
        for idx, layer in enumerate(self.layers):
            name = layer.__name__()
            output_Shape = None
            param = None

            if (idx == 0):
                if layer.input_shape == None:
                    continue
                if name == "Dense":
                    output_shape = f'(None, {layer.units})'
                    param = layer.input_shape[0] * layer.units
                    prev_output_shape = (layer.units, )
                elif name == "Conv2D":
                    k_height, k_width, k_channel = layer.kernel.shape
                    i_height, i_width, i_channel = layer.input_shape
                    o_height = (i_height-k_height+2 *
                                layer.padding)//layer.stride[0]+1
                    o_width = (i_width-k_width+2 *
                               layer.padding)//layer.stride[1]+1
                    output_shape = f'(None, {o_height}, {o_width}, {layer.filters})'
                    param = layer.filters * \
                        (k_height * k_width * i_channel + 1)
                    prev_output_shape = (o_height, o_width, layer.filters)
                elif name == "Pooling":
                    i_height, i_width, i_channel = input_shape
                    o_height = (i_height-layer.size)//layer.stride+1
                    o_width = (i_width-layer.size)//layer.stride+1
                    output_shape = f'(None, {o_height}, {o_width}, {i_channel})'
                    prev_output_shape = (o_height, o_width, i_channel)
                    param = 0
                elif name == "Flatten":
                    i_height, i_width, i_channel = input_shape
                    o_width = i_channel * i_height * i_width
                    output_shape = f'(None, {o_width})'
                    prev_output_shape = (o_width, )
                    param = 0
            else:
                if name == "Dense":
                    output_shape = f'(None, {layer.units})'
                    param = (prev_output_shape[0]+1) * layer.units
                    prev_output_shape = (layer.units, )
                elif name == "Conv2D":
                    k_height, k_width, k_channel = layer.kernel.shape
                    i_height, i_width, i_channel = prev_output_shape
                    o_height = (i_height-k_height+2 *
                                layer.padding)//layer.stride[0]+1
                    o_width = (i_width-k_width+2 *
                               layer.padding)//layer.stride[1]+1
                    output_shape = f'(None, {o_height}, {o_width}, {layer.filters})'
                    prev_output_shape = (o_height, o_width, layer.filters)
                    param = layer.filters * \
                        (k_height * k_width * i_channel + 1)
                elif name == "Pooling":
                    i_height, i_width, i_channel = prev_output_shape
                    o_height = (i_height-layer.size)//layer.stride+1
                    o_width = (i_width-layer.size)//layer.stride+1
                    output_shape = f'(None, {o_height}, {o_width}, {i_channel})'
                    prev_output_shape = (o_height, o_width, i_channel)
                    param = 0
                elif name == "Flatten":
                    i_height, i_width, i_channel = prev_output_shape
                    o_width = i_channel * i_height * i_width
                    output_shape = f'(None, {o_width})'
                    prev_output_shape = (o_width, )
                    param = 0

            table.append([f'{layer.name} ({name})', output_shape, param])
            total_param += param

        print()
        print(tabulate(table, headers=heads, tablefmt="github"))
        print()
        print("Total params:", total_param)
        print("Trainable params:", total_param)
        print("Non-Trainable params:", 0)
