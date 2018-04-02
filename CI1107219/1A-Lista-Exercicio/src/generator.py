import numpy as np
import os

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

def data_1A1(size, dtype='train',verbose=False):
    try:
        X = np.genfromtxt('../data/'+dtype+'/1a1X.txt',delimiter=',')
        Y = np.genfromtxt('../data/'+dtype+'/1a1Y.txt',delimiter=',')
        if verbose:
            print('Inputs: ', X)
            print('Labels: ', Y)
        return X,Y
    except Exception as err:
        if verbose:
            print(err)
        X = np.random.randint(2,size=(size,3))
        Y = pattern_1A1(X)
        # salva arquivo
        np.savetxt('../data/'+dtype+'/1a1X.txt', X, delimiter=',', fmt='%.0f')
        np.savetxt('../data/'+dtype+'/1a1Y.txt', Y, delimiter=',', fmt='%.0f')
        return X,Y

def data_reset_1A1(dtype='train',verbose=False):
    try:
        os.remove('../data/'+dtype+'/1a1X.txt')
        os.remove('../data/'+dtype+'/1a1Y.txt')
        if verbose:
            print('Files removed: 1a1X.txt, 1a1Y.txt')
    except Exception as err:
        if verbose:
            print(err)
            