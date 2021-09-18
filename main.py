import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers

from PIL import Image
from LEMBUT import util

# model = Sequential()
# model.add(layers.Dense(2, "sigmoid", "input"))
# model.add(layers.Dense(3, "sigmoid", "hidden"))

# X = np.random.rand(3, 2)
# y = model(X)
# print(y)
img = util.imgToMat(
    "C:\\Users\\Muslim\\Pictures\\1115214.jpg", size=(224, 224))
kernel = np.array([
    [[9, 7, 1], [1, 3, 1], [3, 4, 1]],
    [[3, 0, 11], [17, 1, 1], [1, 3, 7]],
    [[1, 4, 4], [9, 1, 1], [1, 0, 0]]
])
print(util.conv(img, kernel))
