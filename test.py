from LEMBUT import layers
from LEMBUT import util
import numpy as np
# in_mat = np.array([
#     [[1,2,3,0],
#     [2,3,4,1],
#     [3,4,5,9],
#     [2,3,6,1]],

#     # [[1,2,3,0],
#     # [2,3,4,1],
#     # [3,4,5,9],
#     # [2,3,6,1]],

#     # [[1,2,3,0],
#     # [2,3,4,1],
#     # [3,4,5,9],
#     # [2,3,6,1]]
# ])

in_mat = np.array([
    [[1],[2],[3],[4]],
    [[1],[2],[3],[4]],
    [[1],[2],[3],[4]],
    [[1],[2],[3],[4]]
])

dOut = np.array([
    [[1],[1],[1]],
    [[1],[1],[1]],
    [[1],[1],[1]],
])

dOut2 = np.array([
    [[1,1],[1,1],[1,1]],
    [[1,1],[1,1],[1,1]],
    [[1,1],[1,1],[1,1]],
])

filter = np.array([
    [[1,0],
    [1,0]],
    
    [[0,1],
    [1,0]],

    [[0,1],
    [0,1]]
])

# print(in_mat.shape)
img = util.loadImage('image.jpg', size=(228, 228))

conv_layer = layers.Conv2D(name="conv_1", filters=2, kernel_size=(2, 2), input_shape=(228, 228, 3), padding=1, stride=(2,2))
out = conv_layer.forward(img)
# out = conv_layer.zero_pad(in_mat, 0)
# print(out)
# print(out.shape)
backOut, weightOut = conv_layer.backward(in_mat, dOut2)
print(backOut)
print(weightOut)