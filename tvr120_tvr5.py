# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27
"""
import os
import warnings

import numpy as np
from tqdm import tqdm

from utils import power
from utils import summary
from utils.load_data import *

warnings.filterwarnings('ignore')

alpha_name = 'tvr120_tvr5'
os.makedirs(alpha_name, exist_ok=True)


def process(data: np.ndarray, sig: np.ndarray):
    res = np.full_like(data, np.nan)
    res[sig < 0] = - 1 / data[sig < 0]
    res[sig > 0] = - data[sig > 0]
    return res


alpha = np.full_like(returns.values, np.nan)
for di in tqdm(range(200, alpha.shape[0])):
    adj_wnd = 120
    adj = adjfactor[di - adj_wnd:di]
    adj = adj[::-1].cumprod()[::-1]
    # adj_open = (open_[di - adj_wnd:di] * adj).values
    adj_close = (close[di - adj_wnd:di] * adj).values

    s = negcap[di - adj_wnd:di] / adj_close
    turnover = volume[di - adj_wnd:di] / s
    sig = np.nanmean(turnover, axis=0) / np.nanmean(turnover[-5:], axis=0) - 1
    alpha[di] = sig

alpha[~TOP3000.values] = np.nan
power(alpha)
summary(alpha, alpha_name)
np.save(f'{alpha_name}/{alpha_name}.npy', alpha)
