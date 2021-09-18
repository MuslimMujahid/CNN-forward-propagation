import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2
from LEMBUT import util

img = util.imgToMat("image.jpg", size=(32, 32))

model = Sequential()
model.add(layers.Conv2D("Conv2D", filters=6, kernel_size=(5, 5)))
model.add(layers.Conv2D("Conv2D", filters=16, kernel_size=(5, 5)))
result = model(img)
print(result)
print("result shape", result.shape)
