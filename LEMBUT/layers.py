import numpy as np
from .functions import ACTIVATION_FUNCTIONS
from .util import conv2D


class Layer:
    def __init__(self, name: str, input_shape: tuple, learning_rate: float = 0.01) -> None:
        self.name = name
        self.input_shape = input_shape
        self.learning_rate = learning_rate


class Dense(Layer):
    def __init__(self, units: int, activation: str = "linear", name: str = "dense", input_shape: tuple = None, initial_weight: np.ndarray = None, bias=None, learning_rate: float = 0.01) -> None:
        super().__init__(name, input_shape, learning_rate)
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
            dE_do = None

            if (self.activation == 'softmax'):
                dE_do = y
                label_idx = np.where(y == 1)[0]
                for idx in label_idx:
                    dE_do[idx] = -(1 - dE_do[idx])
            else:
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
    def __init__(self, filters: np.ndarray, name: str, kernel_size: tuple, activation: str, padding: int, stride: tuple, bias: int = 0, input_shape: tuple = None, learning_rate: float = 0.01):
        super().__init__(name, input_shape, learning_rate)
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.filters = filters
        self.kernel_size = kernel_size
        # initialize random kernels
        self.kernel = np.zeros(
            [kernel_size[0], kernel_size[1], filters], dtype=float)
        self.padding = padding
        self.stride = stride
        # self.bias = bias
        self.bias = np.zeros((filters))


""" Convolutional 2D Layer
filter : describes how many filter


"""


class Conv2D(Conv):
    def __init__(self, filters: int, name: str = "conv2d", kernel_size: tuple = (3, 3), activation: str = "relu", padding: int = 0, stride: tuple = (1, 1), bias: int = 0, input_shape: tuple = None, learning_rate: float = 0.01):
        super().__init__(filters, name, kernel_size,
                         activation, padding, stride, bias, input_shape, learning_rate)

    def __name__(self):
        return "Conv2D"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Create buffer for feature maps
        n_sample, x_height, x_width, x_channel = X.shape
        k_height, k_width, k_channel = self.kernel.shape
        f_height = (x_height-k_height+2*self.padding) // self.stride[0] + 1
        f_width = (x_width-k_width+2*self.padding) // self.stride[1] + 1

        lst_feature_maps = []
        for k in range(n_sample):
            feature_maps = np.zeros(
                [f_height, f_width, self.filters], dtype=float)
            # Do convolution for every input channel to each filter channel
            for i in range(self.filters):
                for j in range(x_channel):
                    feature_maps[:, :, j] = np.add(feature_maps[:, :, j], conv2D(
                        X[k, :, :, j], self.kernel[:, :, i], self.padding, self.stride, self.bias))
                feature_maps[:, :, i] += self.bias[i]

            # Detector
            feature_maps = self.activation(feature_maps)

            # insert to list
            lst_feature_maps.append(feature_maps)

        return np.array(lst_feature_maps)

    def backward(self, X: np.ndarray, y: np.ndarray = None, next_layer: Layer = None):
        # legend
        # x : input : X
        # y : output : not used
        # next_layer : layer in front of current layer
        pad = self.padding

        # print("X, prepad", X)
        locX = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        # print("X, postpad", X)
        print(locX.shape)
        stride = self.stride
        in_width, in_height, _ = locX.shape
        stride_x, stride_y = self.stride
        k_height, k_width = self.kernel_size

        # gradient loss of the filters
        dfilt = np.zeros(self.kernel.shape)
        # gradient loss of the biases
        dbias = np.zeros((self.filters))
        # gradient loss of with respect to input (to be propagated to previous layer)
        dout = np.zeros(locX.shape)
        # print("dout shape", dout.shape)

        # loops through all filters
        for f in range(0, self.filters):
            current_y = output_y = 0
            while current_y + k_height <= in_height:
                current_x = output_x = 0
                while current_x + k_width <= in_width:
                    # getting the receptive field of the input, multiplied with constant from output gradient
                    print(current_x, current_x+k_height, current_y,
                          current_y+k_width, (in_height, in_width), y.shape)
                    mult_mat = locX[current_x:current_x+k_height,
                                    current_y:current_y+k_width, 0] * y[output_x, output_y, f]
                    # filter weight updated using the result from above
                    dfilt[:, :, f] += mult_mat

                    # multiply the current filter's kernel with constant from output gradient
                    mult_mat = self.kernel[:, :, f] * y[output_x, output_y, f]
                    # updating the loss gradient with respect to input
                    # this code kinda sus, need recheck
                    dout[current_x:current_x+k_height,
                         current_y:current_y+k_width, 0] += mult_mat

                    # increment iterator
                    current_x += stride_x
                    output_x += 1
                current_y += stride_y
                output_y += 1
            # combine gradient biases based from the output gradient (kinda sus, recheck also)
            dbias[f] = np.sum(y[:, :, f])
        # update bias for current layer
        self.bias += dbias * self.learning_rate
        # return output gradient with respect to input, and filter's loss gradient
        # returned for information to previous layer's backpropagation
        # print("Kernel shape", self.kernel.shape)
        # print("Grad Kernel shape", dfilt.shape)
        # print("Kernel before change")
        # print(self.kernel)
        # print("Kernel after change")
        self.kernel += dfilt * self.learning_rate
        # print(self.kernel)
        return dout, dfilt


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
        n_sample, x_height, x_width, x_channel = X.shape
        lst_outputs = []

        for m in range(n_sample):
            output = np.zeros([(x_height-self.size) //
                              self.stride+1, (x_width-self.size)//self.stride+1, x_channel])
            o_height, o_width, _ = output.shape
            for i in range(x_channel):
                for j in range(o_height):
                    for k in range(o_width):
                        output[j, k, i] = pool_function(
                            X[m, j*self.stride:j*self.stride+self.size, k*self.stride:k*self.stride+self.size, i])

            # insert result
            lst_outputs.append(output)

        return np.array(lst_outputs)


class Flatten(Layer):
    def __init__(self, name: str = "flatten", input_shape: tuple = None) -> None:
        super().__init__(name, input_shape)

    def __name__(self):
        return "Flatten"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = []
        for i in range(X.shape[0]):
            output.append(X[i, :, :, :].flatten('F'))

        return np.array(output)
