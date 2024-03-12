# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27
"""
import numpy as np


def arraycorr(rolwnd1, rolwnd2, axis=1):
    x1 = np.copy(rolwnd1)
    y1 = np.copy(rolwnd2)
    if x1.dtype != np.float64:
        x1 = x1.astype(np.float64)
    idx = np.isnan(x1) | np.isnan(y1)
    x1[idx] = np.nan
    y1[idx] = np.nan
    xy = x1 * y1
    xx = x1 * x1
    yy = y1 * y1
    sx = np.nansum(x1, axis=axis)
    sy = np.nansum(y1, axis=axis)
    sxy = np.nansum(xy, axis=axis)
    sxx = np.nansum(xx, axis=axis)
    syy = np.nansum(yy, axis=axis)
    nn = np.sum((~np.isnan(x1)) & (~np.isnan(y1)), axis=axis)
    val = (nn * sxy - sx * sy) / np.sqrt(nn * sxx - sx * sx) / np.sqrt(nn * syy - sy * sy)
    return val

def arrayrank(rolwnd1, axis=1):
    x1 = np.copy(rolwnd1)
    if x1.dtype != np.float64:
        x1 = x1.astype(np.float64)
    nn = np.sum(np.isnan(x1))
    val =np.argsort(x1).astype(float)
    val[-nn:] = np.nan
    return val
