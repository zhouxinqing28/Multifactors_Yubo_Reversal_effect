# -*- coding: utf-8 -*-

"""
@Author: Yubo TONG
@Date: 2023/11/27d
"""
import pdb
import warnings

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .tools import arraycorr
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def get_coverage(alpha):
    return np.nansum(~np.isnan(alpha), axis=1)


def get_ic(alpha):
    returns = pd.read_pickle('data/returns.pkl')
    alp = alpha[:-1]
    rtn = returns.values[1:]#计算ic
    alp = rankdata(alp,nan_policy="omit",axis=1)
    rtn = rankdata(rtn,nan_policy="omit",axis=1)
    #correlation, p_value = spearmanr(alp, rtn)
    return np.append(arraycorr(alp, rtn), np.nan)
"""def get_ic(alpha):
    returns = pd.read_pickle('data/returns.pkl')
    alp = alpha[:-1]
    rtn = returns.values[1:]#计算ic
    #correlation, p_value = spearmanr(alp, rtn)
    return np.append(arraycorr(alp, rtn), np.nan)"""


def get_group_cumrtn(alphas, ngroups=10):
    ngprtn = np.full((alphas.shape[0], ngroups), np.nan)
    returns = pd.read_pickle('data/returns.pkl').values
    for di in range(alphas.shape[0]):
        rtn = returns[di]
        alpha = alphas[di]
        mask = ~(np.isnan(alpha) | np.isnan(rtn))
        alpha = alpha[mask]
        if alpha.shape[0] > 0:
            rtn_demean = rtn[mask] - np.nanmean(rtn[mask])  # demean
            percentiles = np.percentile(alpha, np.linspace(0, 100, ngroups + 1)[1:])
            groups = np.digitize(alpha, percentiles, right=True)
            group_counts = np.bincount(groups)
            group_weighted_sums = np.bincount(groups, weights=rtn_demean)
            try:
                ngprtn[di] = group_weighted_sums / group_counts
            except:
                pdb.set_trace()
    return ngprtn


def summary(alpha, name):
    cvrg = get_coverage(alpha)
    ic = get_ic(alpha)
    gp_rtn = get_group_cumrtn(alpha)

    #  build dataframe
    df_ngprtn = pd.DataFrame(gp_rtn)
    rtn_b_tot = df_ngprtn.mean()
    df_ngprtn['year'] = [str(_)[:4] for _ in pd.read_pickle('data/returns.pkl').index.tolist()]
    df_ngprtn.dropna(subset=[0], inplace=True)
    rtn_b_year = df_ngprtn.groupby('year').mean()

    df_ic = pd.DataFrame(ic, columns=['ic'])
    df_ic.reset_index(inplace=True)
    df_ic.dropna(subset=['ic'], inplace=True)

    df_cover = pd.DataFrame(cvrg, columns=['cover'])
    df_cover.reset_index(inplace=True)
    # plot figure
    fig = plt.figure(figsize=(18, 18))
    ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=1)
    ax1.plot(df_cover['index'], df_cover['cover'], alpha=0.8)

    ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=1, colspan=1)
    ax2.bar(rtn_b_tot.index, rtn_b_tot)
    plt.title(f'{name}', fontsize=20)

    ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=1, colspan=1)
    rtn_b_year.plot(kind='bar', ax=ax3, legend=False)

    ax4 = plt.subplot2grid((3, 3), (1, 0), rowspan=1, colspan=3)
    ax4.axhline(y=0, color='red', linestyle='--')
    ax4.plot(df_ic['index'], df_ic['ic'], alpha=0.5)
    ax4.plot(df_ic['index'], df_ic['ic'].rolling(window=20).mean())
    ax4.plot(df_ic['index'], df_ic['ic'].rolling(window=120).mean())
    ax4.text(0.05, .95, "Mean %.3f \n Std. %.3f" % (df_ic['ic'].mean(), df_ic['ic'].std()), fontsize=15, transform=ax4.transAxes, verticalalignment='top')

    ax5 = plt.subplot2grid((3, 3), (2, 0), rowspan=1, colspan=3)
    del df_ngprtn['year']
    cumrtn = (1 + df_ngprtn).cumprod()
    cumrtn.plot(ax=ax5)
    long = cumrtn[9]
    win = (long.diff() > 0).sum() / len(long)
    rtn = pow(long.iloc[-1], (250 / len(long))) - 1
    rtn *= 100
    net_value = np.array(long)
    daily_returns = np.diff(net_value) / net_value[:-1]
    sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    cumulative_return = np.cumprod(1 + daily_returns)
    drawdown = 1 - cumulative_return / np.maximum.accumulate(cumulative_return)
    max_drawdown = np.max(drawdown) * 100
    ax5.text(0.05, .95, "LONG ONLY PERFORMANCE \nRtn: %.2f \nWin: %.3f \nshrp: %.2f \nMdd: %.2f" % (rtn, win, sharpe_ratio, max_drawdown), fontsize=15, transform=ax5.transAxes, verticalalignment='top')

    plt.savefig(f'{name}/{name}.jpg', bbox_inches='tight')
    # finish plot
    return
