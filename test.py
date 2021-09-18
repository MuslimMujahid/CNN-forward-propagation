from LEMBUT.layers import Conv
import numpy as np
in_mat = np.array([
    [[1,2,3,0],
    [2,3,4,1],
    [3,4,5,9],
    [2,3,6,1]],

    [[1,2,3,0],
    [2,3,4,1],
    [3,4,5,9],
    [2,3,6,1]],

    [[1,2,3,0],
    [2,3,4,1],
    [3,4,5,9],
    [2,3,6,1]]
])

filter = np.array([
    [[1,0],
    [1,0]],
    
    [[0,1],
    [1,0]],

    [[0,1],
    [0,1]]
])

print(in_mat.shape)

conv_layer = Conv("Test", in_mat.shape, filter, 0, 1)
out = conv_layer.Conv3D(in_mat, filter)
print(out)