import numpy as np


def kth_root(x, k):
    if k % 2 != 0:
        res = np.power(np.abs(x), 1./k)
        return res*np.sign(x)
    else:
        return np.power(np.abs(x), 1./k)


def kth_power(x, k):
    if k % 2 != 0:
        res = np.power(np.abs(x), k)
        return res*np.sign(x)
    else:
        return np.power(np.abs(x), k)
