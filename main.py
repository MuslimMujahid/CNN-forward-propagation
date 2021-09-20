import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2
from LEMBUT import util
from keras.datasets import mnist


(trainX, trainY), (testX, testY) = mnist.load_data()



# img = util.imgToMat("image.jpg", size=(32, 32))
# print(img)
model = Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1)))
model.add(layers.Pooling())
model.add(layers.Conv2D(filters=16, kernel_size=(5, 5)))
model.add(layers.Pooling())
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation="relu"))
model.add(layers.Dense(units=84, activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))
# result = model(img)
for i in range(10):
    img = np.array([trainX[i]])
    # print(img.shape)
    # print(trainX.shape)
    # print(np.ndarray(trainX[i]).shape)
    result = model(img)
    print(result)
    # print("result shape", result.shape)

model.summary()
