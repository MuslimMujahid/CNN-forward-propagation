import numpy as np
from .functions import ACTIVATION_FUNCTIONS
from .util import conv2D


class Layer:
    def __init__(self, name: str, input_shape: tuple) -> None:
        self.name = name
        self.input_shape = input_shape


class Dense(Layer):
    def __init__(self, units: int, activation: str = "linear", name: str = "dense", input_shape: tuple = None, initial_weight: np.ndarray = None, bias=None) -> None:
        super().__init__(name, input_shape)
        self.activation = activation
        self.units = units
        self.W = initial_weight
        self.bias = bias

    def __name__(self):
        return "Dense"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if (self.bias is not None):
            self.input = np.append(X, np.reshape(
                [self.bias for _ in range(X.shape[0])], (X.shape[0], 1)), axis=1)
        else:
            self.input = X

        input_size = self.input.shape[0 if len(self.input.shape) == 1 else 1]

        if self.W is None:
            self.W = np.zeros([input_size, self.units], dtype=float)

        self.net = np.dot(self.input, self.W)
        self.output = ACTIVATION_FUNCTIONS[self.activation](self.net)
        return self.output

    def backward(self, X: np.ndarray, y: np.ndarray = None, next_layer: Layer = None):
        dE_dnet = None

        if y is not None:
            dE_do = -(y - X)
            do_dnet = ACTIVATION_FUNCTIONS['d' + self.activation](X)
            dE_dnet = dE_do * do_dnet
        else:
            dE_do = np.dot(X, next_layer.W.T)
            do_dnet = ACTIVATION_FUNCTIONS['d' +
                                           self.activation](next_layer.input)
            dE_dnet = dE_do * do_dnet

            if (self.bias is not None):
                dE_dnet = dE_dnet[:, :-1]

        # print(self.input.T.shape, dE_dnet.shape)
        dE_dw = np.dot(self.input.T, dE_dnet)

        return dE_dnet, dE_dw


class Conv(Layer):
    def __init__(self, filters: np.ndarray, name: str, kernel_size: tuple, activation: str, padding: int, stride: tuple, bias: int = 0, input_shape: tuple = None):
        super().__init__(name, input_shape)
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.filters = filters
        self.kernel_size = kernel_size
        # initialize random kernels
        self.kernel = np.zeros(
            [kernel_size[0], kernel_size[1], filters], dtype=float)
        self.padding = padding
        self.stride = stride
        self.bias = bias


class Conv2D(Conv):
    def __init__(self, filters: int, name: str = "conv2d", kernel_size: tuple = (3, 3), activation: str = "relu", padding: int = 0, stride: tuple = (1, 1), bias: int = 0, input_shape: tuple = None):
        super().__init__(filters, name, kernel_size,
                         activation, padding, stride, bias, input_shape)

    def __name__(self):
        return "Conv2D"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Create buffer for feature maps
        x_height, x_width, x_channel = X.shape
        k_height, k_width, k_channel = self.kernel.shape
        f_height = (x_height-k_height+2*self.padding) // self.stride[0] + 1
        f_width = (x_width-k_width+2*self.padding) // self.stride[1] + 1
        feature_maps = np.zeros([f_height, f_width, self.filters], dtype=float)

        # Do convolution for every input channel to each filter channel
        for i in range(self.filters):
            for j in range(x_channel):
                feature_maps[:, :, j] = np.add(feature_maps[:, :, j], conv2D(
                    X[:, :, j], self.kernel[:, :, i], self.padding, self.stride, self.bias))

        # Detector
        feature_maps = self.activation(feature_maps)

        return feature_maps

    def backward(self, X: np.ndarray, y: np.ndarray = None, next_layer: Layer = None):
        # legend
        ## x : input
        ## y : output
        ## next_layer : layer in front of current layer

        # backprop for gradient on kernel
        ## matmul between current input with incoming output
        kernel_back = np.matmul(X, y)

        # backprop for gradient on bias
        bias_back = y

        # backprop for gradient on input (for previous layer)
        # full convolution of output gradient with rotated 180 degrees of kernel
        
        
        


class Pooling(Layer):
    def __init__(self, name=None, size=2, stride=2, mode="max", input_shape: tuple = None) -> None:
        super().__init__(name if name is not None else (mode + "_pooling"), input_shape)
        self.size = size
        self.stride = stride
        self.mode = mode

    def __name__(self):
        return "Pooling"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.pool(X)

    def pool(self, X: np.ndarray) -> np.ndarray:
        pool_function = np.mean if self.mode == "avg" else np.max
        x_height, x_width, x_channel = X.shape
        output = np.zeros([(x_height-self.size) //
                          self.stride+1, (x_width-self.size)//self.stride+1, x_channel])
        o_height, o_width, _ = output.shape
        for i in range(x_channel):
            for j in range(o_height):
                for k in range(o_width):
                    output[j, k, i] = pool_function(
                        X[j*self.stride:j*self.stride+self.size, k*self.stride:k*self.stride+self.size, i])

        return output


class Flatten(Layer):
    def __init__(self, name: str = "flatten", input_shape: tuple = None) -> None:
        super().__init__(name, input_shape)

    def __name__(self):
        return "Flatten"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X.flatten('F')
