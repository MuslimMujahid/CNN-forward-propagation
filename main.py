import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2
from LEMBUT import util
# from keras.datasets import mnist


# (trainX, trainY), (testX, testY) = mnist.load_data()

model = Sequential()

# Convolutional layer
# model.add(layers.Conv2D(name="conv_1", filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1)))
# model.add(layers.Pooling(name="pooling_1"))
# model.add(layers.Conv2D(name="conv_2", filters=16, kernel_size=(5, 5)))
# model.add(layers.Pooling(name="pooling_2"))
# model.add(layers.Flatten())

# Fully connected layer
# model.add(layers.Dense(name="dense_1", units=120, activation="relu"))
# model.add(layers.Dense(name="dense_2", units=84, activation="relu"))
# model.add(layers.Dense(name="dense_3", units=10, activation="softmax"))

model.add(layers.Dense(name="dense_1", units=2, activation="sigmoid",
          initial_weight=np.reshape([0.15, 0.25, 0.2, 0.3, 0.35, 0.35], (3, 2)), bias=1, input_shape=(2, )))
model.add(layers.Dense(name="dense_2", units=2, activation="sigmoid",
          initial_weight=np.reshape([0.4, 0.5, 0.45, 0.55, 0.6, 0.6], (3, 2)), bias=1))

# Predict
# for i in range(10):
#     img = trainX[i][..., None]
#     result = model(img)
#     print(result)

# Summary
# model.summary()

# Train
model.fit(
    np.reshape([0.05, 0.1], (1, 2)),
    np.reshape([0.01, 0.99], (1, 2)),
    epochs=100000
)
print(model(np.reshape([0.05, 0.1], (1, 2))))
