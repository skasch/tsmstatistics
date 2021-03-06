import numpy as np

data = np.array([
    [21.36, 10.71, 1238],
    [26.46, 14.8, 3865],
    [17, 18.45, 2232],
    [5.52, 27.31, 236],
    [9.48, 3.58, 13178],
    [5.29, 1.94, 6612],
    [9.18, 3.43, 9281],
    [5.77, 1.29, 8198],
    [11.58, 2.66, 1874],
    [8.54, 1.76, 6111],
    [11.85, 5.96, 9901],
    [13.52, 6.74, 432],
    [99.59, 40.46, 270],
    [73.01, 50.13, 115],
    [4.38, 3.17, 530],
    [11.31, 10.24, 3790],
    [3.57, 0.09, 66],
    [2.81, .56, 2798],
])

print(np.median((data[:,1] / data[:,0]), axis=0))