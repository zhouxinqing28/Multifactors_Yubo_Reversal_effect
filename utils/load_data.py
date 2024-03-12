# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27
"""
import pandas as pd

returns = pd.read_pickle('data/returns.pkl')
high = pd.read_pickle('data/high.pkl')
open_ = pd.read_pickle('data/open_.pkl')
low = pd.read_pickle('data/low.pkl')
close = pd.read_pickle('data/close.pkl')
volume = pd.read_pickle('data/volume.pkl')
amount = pd.read_pickle('data/amount.pkl')
vwap = pd.read_pickle('data/vwap.pkl')
negcap = pd.read_pickle('data/negcap.pkl')
TOP3000 = pd.read_pickle('data/TOP3000.pkl')
adjfactor = pd.read_pickle('data/adjfactor.pkl')
print(open_.shape)
