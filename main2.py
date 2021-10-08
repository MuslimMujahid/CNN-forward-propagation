import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2
from LEMBUT import util
# from keras.datasets import mnist
from LEMBUT.util import *
# from sklearn.model_selection import train_test_split, KFold


# (trainX, trainY), (testX, testY) = mnist.load_data()

# Split Dataset into 90% Train 10% Test
# X = np.concatenate([trainX, testX])
# y = np.concatenate([trainY, testY])
# trainX, trainY, testX, testY = train_test_split(X, y, test_size=0.1)

model = Sequential()

# Convolutional layer
# model.add(layers.Conv2D(name="conv_1", filters=1,
#           kernel_size=(5, 5), input_shape=(28, 28, 1)))
# model.add(layers.Pooling(name="pooling_1"))
# model.add(layers.Conv2D(name="conv_2", filters=16, kernel_size=(5, 5)))
# model.add(layers.Pooling(name="pooling_2"))
# model.add(layers.Flatten())

# Fully connected layer
# model.add(layers.Dense(name="dense_1", units=120, activation="relu"))
# model.add(layers.Dense(name="dense_2", units=84, activation="relu"))
# model.add(layers.Dense(name="dense_3", units=10, activation="softmax"))

# model.add(layers.Dense(name="dense_1", units=2, activation="sigmoid",
#           initial_weight=np.reshape([0.15, 0.25, 0.2, 0.3, 0.35, 0.35], (3, 2)), bias=1, input_shape=(2, )))
# model.add(layers.Dense(name="dense_2", units=2, activation="sigmoid",
#           initial_weight=np.reshape([0.4, 0.5, 0.45, 0.55, 0.6, 0.6], (3, 2)), bias=1))

model.add(layers.Dense(name="dense_1", units=2,
          activation="sigmoid", bias=1, input_shape=(2, )))
model.add(layers.Dense(name="dense_2", units=2, activation="softmax"))

# Predict
# for i in range(10):
#     img = trainX[i][..., None]
#     result = model(img)
#     print(result)

# Summary
model.summary()

# save(model,'test.pkl')
# new_model = load('test.pkl')
# for i in range(10):
#     img = trainX[i][..., None]
#     result = new_model(img)
#     print(result)

# # Summary
# new_model.summary()

# model.summary()

# Train
model.fit(
    np.reshape([0.05, 0.1], (1, 2)),
    np.reshape([0, 1], (1, 2)),
    epochs=1
)
print(model(np.reshape([0.05, 0.1], (1, 2))))
