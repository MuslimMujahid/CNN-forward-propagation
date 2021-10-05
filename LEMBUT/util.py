import numpy as np
import cv2


def loadImage(filepath: str, size=None):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    if (size):
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img


def showImage(img: np.ndarray) -> None:
    cv2.imshow("image", img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def conv2D(X: np.ndarray, kernel: np.ndarray, padding: int, stride: tuple, bias: int) -> np.ndarray:
    # padding
    X = np.pad(X, padding, mode='constant')
    x_height, x_width = X.shape
    k_height, k_width = kernel.shape

    f_height, f_width = ((x_height-k_height) //
                         stride[0] + 1, (x_width-k_width)//stride[1] + 1)
    feature_map = np.zeros([f_height, f_width], dtype=int)

    for i in range(f_height):
        for j in range(f_width):
            feature_map[i, j] = np.sum(np.multiply(X[i*stride[1]:i*stride[1]+k_width,
                                                     j*stride[0]:j*stride[0]+k_width], kernel[:, :]))

    return feature_map


def conv3D(X: np.ndarray, kernel: np.ndarray, padding: int, stride: tuple, bias: int) -> np.ndarray:
    return np.add(
        conv2D(X[2, :, :], kernel[2, :, :], padding, stride, bias),
        np.add(
            conv2D(X[0, :, :], kernel[0, :, :], padding, stride, bias),
            conv2D(X[1, :, :], kernel[1, :, :], padding, stride, bias)))

    return feature_map
