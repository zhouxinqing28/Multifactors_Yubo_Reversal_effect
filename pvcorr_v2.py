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
from utils import arraycorr
from utils.load_data import *

warnings.filterwarnings('ignore')

alpha_name = 'pvcorr_v2'
os.makedirs(alpha_name, exist_ok=True)

alpha = np.full_like(returns.values, np.nan)
for di in tqdm(range(200, alpha.shape[0])):
    adj_wnd = 10
    adj = adjfactor[di - adj_wnd:di]
    adj = adj[::-1].cumprod()[::-1]
    # adj_open = (open_[di - adj_wnd:di] * adj).values
    # adj_close = (close[di - adj_wnd:di]).values
    adj_high = (high[di - adj_wnd:di] * adj).values
    # adj_low = (low[di - adj_wnd:di] * adj).values
    vlm = volume[di - adj_wnd:di].values
    alpha[di] = arraycorr(adj_high, vlm, axis=0)

alpha *= -1
alpha[~TOP3000.values] = np.nan
power(alpha)
summary(alpha, alpha_name)
np.save(f'{alpha_name}/{alpha_name}.npy', alpha)
