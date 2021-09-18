import numpy as np
from .functions import ACTIVATION_FUNCTIONS
# from .util import conv3D
from .util import conv2D


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


class Conv(Layer):
    def __init__(self, name: str, filters: np.ndarray, kernel_size: tuple, padding: int, stride: tuple, bias: int = 0):
        super().__init__("linear", name)
        self.filters = filters
        self.kernel_size = kernel_size
        # self.kernel = np.zeros(
        #     [filters, kernel_size[0], kernel_size[1]], dtype=int)
        self.kernel = np.random.rand(filters, kernel_size[0], kernel_size[1])
        self.padding = padding
        self.stride = stride
        self.bias = bias

class Conv2D(Conv):
    def __init__(self, name: str, filters: int, kernel_size: tuple = (3, 3), padding: int = 0, stride: tuple = (1, 1), bias: int = 0):
        super().__init__(name, filters, kernel_size, padding, stride, bias)

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
        return feature_maps


class Pooling(Layer):
    def __init__(self, units: int, activation: str, name: str, size=2, stride=2) -> None:
        super().__init__(activation, name)
        self.units = units
        self.size = size
        self.stride = stride

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.pool(X)

    def pool(self, X: np.ndarray) -> np.ndarray:
        output = np.zeros((np.uint16((X.shape[0]-self.size+1)/self.stride+1), np.uint16(
            (X.shape[1]-self.size+1)/self.stride+1), X.shape[-1]))
        for map_num in range(X.shape[-1]):
            r2 = 0
            for r in np.arange(0, X.shape[0]-self.size+1, self.stride):
                c2 = 0
                for c in np.arange(0, X.shape[1]-self.size+1, self.stride):
                    output[r2, c2, map_num] = np.max(
                        [X[r:r+self.size,  c:c+self.size]])
                    c2 = c2 + 1
                r2 = r2 + 1
        return output
