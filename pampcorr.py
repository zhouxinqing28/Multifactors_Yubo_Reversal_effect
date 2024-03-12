# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27
"""
import os
import warnings

import numpy as np
from tqdm import tqdm

from utils import arraycorr
from utils import power
from utils import summary
from utils.load_data import *

warnings.filterwarnings('ignore')

alpha_name = 'pampcorr'
os.makedirs(alpha_name, exist_ok=True)

alpha = np.full_like(returns.values, np.nan)
for di in tqdm(range(200, alpha.shape[0])):
    adj_wnd = 10
    amp = high[di - adj_wnd:di] / low[di - adj_wnd:di] - 1
    adj = adjfactor[di - adj_wnd:di]
    adj = adj[::-1].cumprod()[::-1]
    adj_close = (close[di - adj_wnd:di] * adj).values

    alpha[di] = arraycorr(adj_close, amp, axis=0)

alpha *= -1
alpha[~TOP3000.values] = np.nan
power(alpha)
summary(alpha, alpha_name)
np.save(f'{alpha_name}/{alpha_name}.npy', alpha)
