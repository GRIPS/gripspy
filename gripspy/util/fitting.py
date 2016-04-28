from __future__ import division, print_function

import numpy as np

_doughnut = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)

def parapeak(data):
    """
    FIXME
    """
    if data.shape != (3, 3):
        raise ValueError("Input array must be 3x3")

    if np.max(data[_doughnut]) >= data[1, 1]:
        raise ValueError("Central element is not the local maximum")

    if np.prod(2 * data[1, :] - data[0, :] - data[2, :]) == 0:
        raise ValueError("Input array has an issue of the first type")

    x = 0.5 * (data[2, :] - data[0, :]) / (2 * data[1, :] - data[0, :] - data[2, :])
    zx = data[1, :] + 0.25 * x * (data[2, :] - data[0, :])

    xpk = np.mean(x)

    if np.abs(2 * x[1] - x[0] - x[2]) >= 1:
        raise ValueError("Input array has an issue of the second type")

    ypk = 0.5 * (zx[2] - zx[0]) / (2 * zx[1] - zx[0] - zx[2])
    zpk = zx[1] + 0.25 * ypk * (zx[2] - zx[0])

    return xpk, ypk
