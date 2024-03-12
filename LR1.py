import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.impute import SimpleImputer
from utils.load_data import *
from utils import summary
from tqdm import trange,tqdm
import os
from scipy.stats import zscore
from sklearn import preprocessing
# Load the factors
alpha_lst = ['oc5d', 'oc20d', 'overnight5d', 'overnight20d', 'pvcorr_v1', 'pvcorr_v2', 'tvr120_tvr5', 'vol_rtn', 'pampcorr']
names = {}

# 假设所有数据的格式都与open_一致
all_stk_ls = []
for a in tqdm(alpha_lst):
    alpha_df = pd.DataFrame(np.load(f'{a}/{a}.npy'))
    #alpha_df = alpha_df.apply(zscore, axis=1, nan_policy='propagate')有空值无法使用apply；先转化为np再转化回来
    values = alpha_df.values.astype(float)
    data = preprocessing.scale(values)
    df = pd.DataFrame(data)  # 将array还原为dataframe
    names[a] = alpha_df.stack()
    all_stk_ls = alpha_df.columns

#print(names.items())
ret = open_.pct_change(1).shift(-2).reset_index(drop=True).T.reset_index(drop=True).T
names['target'] = ret.stack()

data = pd.concat(names, axis=1).dropna(how='any').unstack()

train = 0
model = None
signal = []#np.full(np.nan, open_.shape)

for i in trange(300*1,data.shape[0]):
    if train % 30 == 0: # 每30天重新训练
        tmp = data.iloc[i-300*1:i].stack()
        X = tmp.loc[:, alpha_lst].values
        y = tmp.loc[:, ['target']].values
        model = LinearRegression()
        model.fit(X, y)
    # 每天正常预测
    valid_stks = data.iloc[[i]].stack().loc[:, alpha_lst].index.get_level_values(1)
    x = data.iloc[[i]].stack().loc[:, alpha_lst].values

    y_pred = pd.Series(model.predict(x).reshape((1,-1))[0],index=valid_stks)
    y_pred = y_pred.reindex(all_stk_ls)
    signal.append(y_pred)

    train += 1
signal = pd.DataFrame(signal)
print(signal.shape)
factor_pre = pd.DataFrame(np.nan, index=range(521), columns=range(4000))
factor_pre = pd.concat([factor_pre,signal], axis=0)
factor_pre = np.array(factor_pre)
#print(factor_pre.shape)
# signal = np.concatenate(signal,axis=0)
alpha_name='LR1'
os.makedirs(alpha_name, exist_ok=True)
summary(factor_pre, alpha_name)

np.save(f'{alpha_name}/{alpha_name}.npy', factor_pre)
#
# # Combine factors
# # alpha_combined = np.nanmean(tot_alpha, axis=2)
# # alpha_combined -= np.nanmean(alpha_combined, axis=1).reshape(alpha_combined.shape[0], -1)
#
# '''
# print(names)
# # Find rows with missing values
# row_mask = np.isnan(names).any(axis=1)
#
# # Remove rows with missing values
# X_cleaned = names[~row_mask]
# y_cleaned = names[~row_mask]
# '''
# # Prepare data for linear regression
# X = alpha_combined[:, :-1]  # Input factors
# y = alpha_combined[:, -1]  # Target factor
#
# # Find columns with any missing values
# col_mask = np.isnan(X).any(axis=0)
#
# # Remove columns with missing values
# X_cleaned = X[:, ~col_mask]
# y_cleaned = y
#
# # Impute missing values in X_cleaned using mean imputation
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X_cleaned)
# """"# Create an imputer object with the desired strategy (e.g., mean)
# imputer = SimpleImputer(strategy='mean')
#
# # Impute the missing values in X
# X_imputed = imputer.fit_transform(X)
# """
# # Create and train the linear regression model
# model = LinearRegression()
# model.fit(X, y)
#
# # Predict the target factor for a new set of factors
# new_factors = np.array([[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]])  # Example new factors
# predicted_factor = model.predict(new_factors)
# print("Predicted factor:", predicted_factor)
