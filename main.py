import numpy as np
import LEMBUT
from LEMBUT.model import Sequential
from LEMBUT import layers
import cv2
from LEMBUT import util
from keras.datasets import mnist
from LEMBUT.util import *
from sklearn.model_selection import train_test_split, KFold

option = input("Choose by typing the number:\n1. Load file\n2. Run new\n")
model = None
if option == "2":
    (trainX, trainY), (testX, testY) = mnist.load_data()

    # Split Dataset into 90% Train 10% Test
    X = np.concatenate([trainX, testX])
    X = X[:, :, :, np.newaxis]
    y = np.concatenate([trainY, testY])
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.1)
    
    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    # print(trainX[0].shape, testX[0].shape)
    # print(len(trainX), len(testX))

    model = Sequential()
    # Convolutional layer
    model.add(layers.Conv2D(name="conv_1", filters=1, kernel_size=(5, 5), input_shape=(28, 28, 1)))
    model.add(layers.Pooling(name="pooling_1"))
    model.add(layers.Conv2D(name="conv_2", filters=16, kernel_size=(5, 5)))
    model.add(layers.Pooling(name="pooling_2"))
    model.add(layers.Flatten())
    # Fully connected layer
    # model.add(layers.Dense(name="dense_1", units=120, activation="relu"))
    # model.add(layers.Dense(name="dense_2", units=84, activation="relu"))
    # model.add(layers.Dense(name="dense_3", units=10, activation="softmax"))

    # model.add(layers.Dense(name="dense_1", units=2, activation="sigmoid",
    #           initial_weight=np.reshape([0.15, 0.25, 0.2, 0.3, 0.35, 0.35], (3, 2)), bias=1, input_shape=(2, )))
    # model.add(layers.Dense(name="dense_2", units=2, activation="sigmoid",
    #           initial_weight=np.reshape([0.4, 0.5, 0.45, 0.55, 0.6, 0.6], (3, 2)), bias=1))

    # Predict
    # for i in range(10):
    #     img = trainX[i][..., None]
    #     result = model(img)
    #     print(result)
else:
    filename = input("Type filename here: ")
    model = util.load(filename)

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
# model.fit(
#     # np.reshape([0.05, 0.1], (1, 2)),
#     # np.reshape([0, 1], (1, 2)),
#     trainX, trainY,
#     epochs=1
# )
# print(model(np.reshape([0.05, 0.1], (1, 2))))

# 10 Fold Cross Validation
k_fold = KFold(n_splits=10)

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    prediction_label = []
    for i in range(prediction.shape[1]):
        prediction_label.append(np.argmax(prediction[:, i]))

    y_test_label = []
    for i in range(y_test.shape[0]):
        y_test_label.append(np.argmax(y_test[i, :]))

    print("Prediction")
    print(prediction_label)
    print("Comparison")
    print(y_test_label)
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test_label, prediction_label))
    print("Precision:")
    precision_result(y_test_label, prediction_label)
    print("Recall:")
    recall_result(y_test_label, prediction_label)
    print("F1:")
    f1_result(y_test_label, prediction_label)
    print("Accuracy : " + str(acc_result(y_test_label, prediction_label)))

    save_choice = input("Want to save the model? (Yes/No)\n")
    if save_choice == 'Yes':
        save_filename = input("Enter filename to save here (format: [filename].pkl)\n")
        util.save(model, save_filename)