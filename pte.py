import numpy as np

in_mat = np.array([
    [[1],[2],[3],[4]],
    [[1],[2],[3],[4]],
    [[1],[2],[3],[4]],
    [[1],[2],[3],[4]]
])

in_mat = np.pad(in_mat, ((1,1),(1,1),(0,0)), mode='constant')
print(in_mat)
print(in_mat.shape)
