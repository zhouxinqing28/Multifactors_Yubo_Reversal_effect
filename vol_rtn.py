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

alpha_name = 'vol_rtn'
os.makedirs(alpha_name, exist_ok=True)

alpha = np.full_like(returns.values, np.nan)
for di in tqdm(range(200, alpha.shape[0])):
    vol = volume[di - 10:di]
    rtn = returns[di - 10:di]
    alpha[di] = vol.iloc[-1] / vol.iloc[-2, :] * rtn.iloc[- 1, :]
alpha *= -1
alpha = pd.DataFrame(alpha).rolling(5).mean().values

percentile_20 = np.nanpercentile(alpha, 20, axis=1)
percentile_80 = np.nanpercentile(alpha, 80, axis=1)
alpha = np.where((alpha >= percentile_20[:, None]) & (alpha <= percentile_80[:, None]), alpha, np.nan)

alpha[~TOP3000.values] = np.nan
power(alpha)
summary(alpha, alpha_name)
np.save(f'{alpha_name}/{alpha_name}.npy', alpha)
