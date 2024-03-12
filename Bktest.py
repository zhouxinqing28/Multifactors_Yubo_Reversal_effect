# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/28
"""
import os
import pdb
import warnings
import numpy as np
from tqdm import tqdm
from utils import power
from utils import summary
from utils.load_data import *

warnings.filterwarnings('ignore')

# load_alphas
# alpha_lst = ['oc5d', 'oc20d', 'overnight5d', 'overnight20d', 'pvcorr_v1', 'pvcorr_v2', 'tvr120_tvr5', 'vol_rtn', 'pampcorr']
alpha_lst = [ 'oc20d', 'overnight5d',  'pvcorr_v2', 'tvr120_tvr5', 'vol_rtn','pampcorr']

names = locals()
for a in alpha_lst:
    names[a] = np.load(f'{a}/{a}.npy')
tot_alpha = np.stack([names[i] for i in alpha_lst], axis=2)

alpha_name = 'cmb'
os.makedirs(alpha_name, exist_ok=True)
alpha = np.nanmean(tot_alpha, axis=2)
alpha -= np.nanmean(alpha, axis=1).reshape(alpha.shape[0], -1)
summary(alpha, alpha_name)
