import numpy as np
import cv2


def imgToMat(filepath: str, size=None):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if (size):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img


def conv(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    x_channel, x_height, x_width = X.shape
    k_channel, k_height, k_width = kernel.shape

    f_height, f_width = (x_width - k_width + 1, x_height - k_height + 1)
    feature_map = np.zeros(shape=(f_height, f_width))
    for i in range(f_height):
        for j in range(f_width):
            result = 0
            for k in range(x_channel):
                result += np.sum(np.multiply(X[k, i:i+k_width,
                                               j:j+k_width], kernel[k, :, :]))
            feature_map[i, j] = result

    return feature_map
