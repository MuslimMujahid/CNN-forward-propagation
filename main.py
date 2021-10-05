import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2
from LEMBUT import util
from keras.datasets import mnist


(trainX, trainY), (testX, testY) = mnist.load_data()

model = Sequential()

# Convolutional layer
model.add(layers.Conv2D(name="conv_1", filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1)))
model.add(layers.Pooling(name="pooling_1"))
model.add(layers.Conv2D(name="conv_2", filters=16, kernel_size=(5, 5)))
model.add(layers.Pooling(name="pooling_2"))
model.add(layers.Flatten())

# Fully connected layer
model.add(layers.Dense(name="dense_1", units=120, activation="relu"))
model.add(layers.Dense(name="dense_2", units=84, activation="relu"))
model.add(layers.Dense(name="dense_3", units=10, activation="softmax"))

# Predict
for i in range(10):
    img = trainX[i][..., None]
    result = model(img)
    print(result)

# Summary
model.summary()
