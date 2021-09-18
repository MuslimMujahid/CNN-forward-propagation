import numpy as np
from PIL import Image


def imgToMat(filepath: str):
    f = Image.open(filepath, "r")
    img = np.array(f)
    return img


def conv(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    x_height, x_width, x_channel = X.shape
    k_height, k_width, k_channel = kernel.shape

    f_height, f_width = (x_width - k_width + 1, x_height - k_height + 1)
    feature_map = np.zeros(shape=(f_height, f_width))
    for i in range(f_height):
        for j in range(f_width):
            result = 0
            for k in range(x_channel):
                result += np.sum(np.multiply(X[i:i+k_width,
                                               j:j+k_width, k], kernel[:, :, k]))
            feature_map[i, j] = result

    return feature_map
