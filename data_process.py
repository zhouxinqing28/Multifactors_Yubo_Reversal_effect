# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
# load basic data
di = np.memmap('Basedata/Dates.npy', mode='r', dtype=np.int64)
ii = np.memmap('Basedata/Instruments.npy', mode='r', dtype=np.dtype('U32'))
num2date = {i: j for i, j in enumerate(di)}
num2inst = {i: j for i, j in enumerate(ii)}
date2num = {j: i for i, j in num2date.items()}
inst2num = {j: i for i, j in num2inst.items()}
# load features, here we use open_ high low close volume tvr return some market basedata
returns = pd.DataFrame(np.memmap('Basedata/returns.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di[:4271], columns=ii)
high = pd.DataFrame(np.memmap('Basedata/high.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
open_ = pd.DataFrame(np.memmap('Basedata/open.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
low = pd.DataFrame(np.memmap('Basedata/low.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
close = pd.DataFrame(np.memmap('Basedata/close.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
volume = pd.DataFrame(np.memmap('Basedata/volume.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
amount = pd.DataFrame(np.memmap('Basedata/amount.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
vwap = pd.DataFrame(np.memmap('Basedata/vwap.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
negcap = pd.DataFrame(np.memmap('Basedata/negcap.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)
TOP3000 = pd.DataFrame(np.memmap('Basedata/TOP3000.npy', mode='r', dtype=np.bool).reshape(-1, 5227), index=di, columns=ii)
adjfactor = pd.DataFrame(np.memmap('Basedata/adjfactor.npy', mode='r', dtype=np.float64).reshape(-1, 5227), index=di, columns=ii)

returns = returns.iloc[2400:4265, :4000]
high = high.iloc[2400:4265, :4000]
open_ = open_.iloc[2400:4265, :4000]
low = low.iloc[2400:4265, :4000]
close = close.iloc[2400:4265, :4000]
volume = volume.iloc[2400:4265, :4000]
amount = amount.iloc[2400:4265, :4000]
vwap = vwap.iloc[2400:4265, :4000]
negcap = negcap.iloc[2400:4265, :4000]
TOP3000 = TOP3000.iloc[2400:4265, :4000]
adjfactor = adjfactor.iloc[2400:4265, :4000]

returns.to_pickle('data/returns.pkl')
high.to_pickle('data/high.pkl')
open_.to_pickle('data/open_.pkl')
low.to_pickle('data/low.pkl')
close.to_pickle('data/close.pkl')
volume.to_pickle('data/volume.pkl')
amount.to_pickle('data/amount.pkl')
vwap.to_pickle('data/vwap.pkl')
negcap.to_pickle('data/negcap.pkl')
TOP3000.to_pickle('data/TOP3000.pkl')
adjfactor.to_pickle('data/adjfactor.pkl')
