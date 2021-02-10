import numpy as np

import giwerm.constants as C
from giwerm.utils import kth_power, kth_root


def alpha_geodesic(a, b, lmd, alpha):
    if alpha < C.EPS and 0 in a or 0 in b:
        return np.zeros(a.shape)

    if alpha == -1:
        return np.exp((1 - lmd) * np.log(a) + lmd * np.log(b))
    elif alpha == 1:
        return (1 - lmd) * a + lmd * b
    elif alpha == 0:
        return ((1 - lmd) * np.sqrt(a) + lmd * np.sqrt(b)) / 4
    elif alpha == 3:
        return 1 / ((1 - lmd) / a + lmd / b)

    return kth_root((1 - lmd) * kth_power(a, alpha) + lmd * kth_power(b, alpha), alpha)
