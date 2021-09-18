import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2

from LEMBUT import util

img = util.imgToMat("image.jpg", size=(500, 500))

model = Sequential()

# Test layer 2D
# kernel = np.array([
#     1, 1, 1,
#     0, 0, 0,
#     -1, -1, -1
# ]).reshape((3, 3))
# model.add(layers.Conv2D("Conv2D", kernel, 0, (1, 1), 0))
# res = model(img[:, :, 0])

# Test layer 3D
kernel = np.array([
    1, 1, 1,
    0, 0, 0,
    -1, -1, -1,

    1, 1, 1,
    0, 0, 0,
    -1, -1, -1,

    1, 1, 1,
    0, 0, 0,
    -1, -1, -1
]).reshape((3, 3, 3))
model.add(layers.Conv3D(name="Conv3D", filter=kernel,
          padding=100, stride=(1, 1), bias=0))
res = model(img)

cv2.imshow("image", res.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
