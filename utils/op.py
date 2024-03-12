# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27
"""

import numpy as np
from scipy.stats import rankdata as rd


def rank(x, method='average'):
    iimap = ~np.isnan(x)
    y = x[iimap]
    n = len(y)
    if n == 0:
        return
    elif n == 1:
        y[0] = 0.5
    else:
        y = (rd(y, method) - 1) / float(n - 1)
    x[iimap] = y[:]


def rank2(x, method='average', axis=0):
    if x.ndim != 2:
        raise ValueError('rank2: data is not a matrix')
    if axis == 0:
        for k in range(x.shape[1]):
            rank(x[:, k], method=method)
    else:
        for k in range(x.shape[0]):
            rank(x[k], method=method)


def power(x, exp=3, dorank=True):
    if dorank:
        rank2(x, axis=1)
        x -= 0.5
    x[:] = np.sign(x) * np.power(np.fabs(x), exp)
