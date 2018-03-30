import numpy as np

def pattern_1A1(X):
    Y = []
    for x_i in X:
        if np.array_equal(np.array([0,0,0]), x_i):
            Y.append([1,0,0,0,0,0,0,0])
        elif np.array_equal(np.array([0,0,1]), x_i):
            Y.append([0,1,0,0,0,0,0,0])
        elif np.array_equal(np.array([0,1,0]), x_i):
            Y.append([0,0,1,0,0,0,0,0])
        elif np.array_equal(np.array([0,1,1]), x_i):
            Y.append([0,0,0,1,0,0,0,0])
        elif np.array_equal(np.array([1,0,0]), x_i):
            Y.append([0,0,0,0,1,0,0,0])
        elif np.array_equal(np.array([1,0,1]), x_i):
            Y.append([0,0,0,0,0,1,0,0])
        elif np.array_equal(np.array([1,1,0]), x_i):
            Y.append([0,0,0,0,0,0,1,0])
        elif np.array_equal(np.array([1,1,1]), x_i):
            Y.append([0,0,0,0,0,0,0,1])
    return np.array(Y)

def data_1A1(size, verbose=False):
    X =  np.random.randint(2,size=(size,3))
    Y = pattern_1A1(X)
    return X,Y