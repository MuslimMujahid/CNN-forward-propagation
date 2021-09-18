import numpy as np
from .functions import ACTIVATION_FUNCTIONS
from .util import conv2D


class Layer:
    def __init__(self, name: str) -> None:
        self.name = name


class Dense(Layer):
    def __init__(self, units: int, activation: str, name: str = "dense") -> None:
        super().__init__(name)
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.units = units
        self.W = None

    def __name__(self):
        return "Dense"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        input_size = X.shape[0 if len(X.shape) == 1 else 1]
        if (not self.W):
            self.W = np.random.rand(input_size, self.units)

        net = np.dot(X, self.W)
        output = self.activation(net)

        # For summary
        self.output_shape = f'(None, {output.shape[0]})'
        self.param = (input_size+1)*self.units

        return output


class Conv(Layer):
    def __init__(self, filters: np.ndarray, name: str, kernel_size: tuple, activation: str, padding: int, stride: tuple, bias: int = 0):
        super().__init__(name)
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.filters = filters
        self.kernel_size = kernel_size
        # self.kernel = np.zeros(
        #     [filters, kernel_size[0], kernel_size[1]], dtype=int)
        self.kernel = np.random.rand(filters, kernel_size[0], kernel_size[1])
        self.padding = padding
        self.stride = stride
        self.bias = bias


class Conv2D(Conv):
    def __init__(self, filters: int, name: str = "conv2d", kernel_size: tuple = (3, 3), activation: str = "relu", padding: int = 0, stride: tuple = (1, 1), bias: int = 0):
        super().__init__(filters, name, kernel_size, activation, padding, stride, bias)

    def __name__(self):
        return "Conv2D"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Create buffer for feature maps
        x_channel, x_height, x_width = X.shape
        k_channel, k_height, k_width = self.kernel.shape
        f_height, f_width = ((x_height-k_height+2*self.padding) //
                             self.stride[0] + 1, (x_width-k_width+2*self.padding)//self.stride[1] + 1)
        feature_maps = np.zeros([self.filters, f_height, f_width], dtype=int)

        # Do convolution for every input channel to each filter channel
        for i in range(self.filters):
            for j in range(x_channel):
                feature_maps[j, :, :] = np.add(feature_maps[j, :, :], conv2D(
                    X[j, :, :], self.kernel[i, :, :], self.padding, self.stride, self.bias))

        # Detector
        feature_maps = ACTIVATION_FUNCTIONS["relu"](feature_maps)

        # For summary
        self.output_shape = f'(None, {feature_maps.shape[1]}, {feature_maps.shape[2]}, {feature_maps.shape[0]})'
        self.param = self.filters * \
            (self.kernel_size[0]*self.kernel_size[1]*x_channel+1)

        return feature_maps


class Pooling(Layer):
    def __init__(self, name=None, size=2, stride=2, mode="max") -> None:
        super().__init__(name if name is not None else (mode + "_pooling"))
        self.size = size
        self.stride = stride
        self.mode = mode

    def __name__(self):
        return "Pooling"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.pool(X)

    def pool(self, X: np.ndarray) -> np.ndarray:
        pool_function = np.mean if self.mode == "avg" else np.max
        x_channel, x_height, x_width = X.shape
        output = np.zeros([x_channel, (x_height-self.size) //
                          self.stride+1, (x_width-self.size)//self.stride+1])
        _, o_height, o_width = output.shape
        for i in range(x_channel):
            for j in range(o_height):
                for k in range(o_width):
                    output[i, j, k] = pool_function(
                        X[i, j*self.stride:j*self.stride+self.size, k*self.stride:k*self.stride+self.size])

        # For summary
        self.output_shape = f'(None, {output.shape[1]}, {output.shape[2]}, {output.shape[0]})'
        self.param = 0

        return output


class Flatten(Layer):
    def __init__(self, name: str = "flatten") -> None:
        super().__init__(name)

    def __name__(self):
        return "Flatten"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        flatten = X.flatten('F')

        # For summary
        self.output_shape = f'(None, {flatten.shape[0]})'
        self.param = 0

        return flatten
