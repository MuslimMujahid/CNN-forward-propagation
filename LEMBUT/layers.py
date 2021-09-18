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


class Conv(Layer):
    def __init__(self, name: str, size: tuple, filter: np.ndarray, padding: int, stride: int, bias: int = 0, debug=False):
        super().__init__("linear", name)
        (filter_z, filter_y, filter_x) = filter.shape
        assert filter_z == 3, "Filter is not 3-channeled. Filter shoudl have shape (channel, height, width)"
        self.size = size
        self.filter = filter
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.debug = debug
    
    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        return self.Conv3D(input_array, self.filter)
    
    def Conv2D(self, image: np.ndarray):
        (filter_x, filter_y) = self.filter.shape
        image_dim, in_dim = image.shape
        if self.debug:
            print("filter shape ", self.filter.shape)
            print("image shape ", image.shape)
            print("image dim", image_dim)
            print("in dim", in_dim)

        out_dimension = int((in_dim - filter_x)/self.stride) + 1
        out_mat = np.zeros((out_dimension, out_dimension))
        if self.debug:
            print(out_mat)

        for curr_f in range(filter_x):
            in_y = out_y = 0
            while in_y + filter_y <= in_dim:
                in_x = out_x = 0
                if self.debug: print("y", in_y, out_y)
                while in_x + filter_x <= in_dim:
                    if self.debug:
                        print("============conv==========")
                        print("x", in_x, out_x)
                        print(image[in_y:in_y+filter_y, in_x:in_x+filter_x])
                        print(self.filter)
                    out_mat[out_x, out_y] = np.sum(self.filter * image[in_y:in_y+filter_y, in_x:in_x+filter_x]) + self.bias
                    in_x += self.stride
                    if self.debug: print(out_mat[out_x, out_y])
                    out_x += 1
                in_y += self.stride
                out_y += 1
        
        return out_mat

    def Conv3D(self, image:np.ndarray, filter: np.ndarray):
        (filter_z_dim, filter_y_dim, filter_x_dim) = filter.shape
        (image_z_dim, image_y_dim, image_x_dim) = image.shape
        assert filter_z_dim == 3, "Filter must be 3-channeled"
        out_dimension = int((image_y_dim - filter_y_dim)/self.stride) + 1
        out_mat = np.zeros((filter_z_dim, out_dimension, out_dimension))
        for selected_filter_channel in range(0,filter_z_dim):
            self.filter = filter[selected_filter_channel]
            # print(image[selected_filter_channel])
            out_mat[selected_filter_channel] = self.Conv2D(image[selected_filter_channel])
        return out_mat

    

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
