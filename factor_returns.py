# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from datetime import datetime
from WindPy import w
w.start()

def alpha_delete(alpha_corr, alpha_IR, corr_limit, ir_limit):
    alpha_corr_abs = alpha_corr.abs()
    alpha_list = alpha_corr_abs.columns
    alpha_IR_period = alpha_IR.abs()

    for alpha_i in alpha_list:
        for alpha_j in alpha_list:
            if alpha_i == alpha_j:
                continue

            if alpha_i in alpha_corr_abs.columns and alpha_j in alpha_corr_abs.columns:

                if alpha_corr_abs.loc[alpha_i, alpha_j] > corr_limit:
                    to_delete = alpha_IR_period.ix[[alpha_i, alpha_j]].argmin()
                    alpha_corr_abs = alpha_corr_abs.drop(to_delete, axis=1)
                    alpha_corr_abs = alpha_corr_abs.drop(to_delete, axis=0)

    IR = alpha_IR_period.ix[alpha_corr_abs.index]
    IR = IR[IR > ir_limit].dropna()
    return IR


if __name__ == '__main__':

    dt = datetime.now()

    if dt.hour < 16:
        date_before = w.tdaysoffset(-1, dt).Data[0][0]
        date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
        dt = date_before

    period = 2
    alpha_corr = pd.read_hdf('E:/multi_factor/basic_factor/alpha_corr.h5','table')
    ic_2 = pd.read_hdf('E:/multi_factor/adjusted_ic/ic_2.h5','table')
    price_data = pd.read_hdf('E:\multi_factor\price_data\price_data.h5','table')
    volume = price_data['volume']
    returns = price_data['adjopen'].pct_change(period).shift(-period - 1)
    cap = price_data['close'] * price_data['total_shares']
    alpha_IR = (ic_2.mean() / ic_2.std() * np.sqrt(252.0)).abs()
    ir = alpha_delete(alpha_corr, alpha_IR, 0.4, 0.0)
    factor_returns = pd.read_hdf('E:/multi_factor//factor_returns/factor_returns.h5','table')
    if factor_returns.empty:
        begin_date = '2007-01-31'
    else:
        begin_date = factor_returns.index[-1]
    tradedate = w.tdays(begin_date, dt).Data[0]
    tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)
    choose = u'有色金属'
    for date in tradedate[:-period - 1]:
        print date

        barra_factor = pd.read_hdf('E:/multi_factor/barra_factor/%s.h5' % date, 'table')
        alpha_factor = pd.read_hdf('E:/multi_factor/alpha_factor/%s.h5' % date, 'table')
        columns = list(set(ir.index) & set(alpha_factor.columns))
        columns.sort()
        alpha_factor = alpha_factor[columns]

        volume_t = volume.ix[date]
        returns_t = returns.ix[date]
        factor = pd.concat([barra_factor, alpha_factor], axis=1)
        factor = factor[volume_t > 0]
        returns_t = returns_t.ix[factor.index].fillna(0)
        cap_t = cap.ix[date].ix[factor.index].fillna(cap.ix[date].quantile())

        industry_key = barra_factor.columns[10:-1]
        industry_set = factor[industry_key]
        industry_cap = pd.Series()
        for industry_name in industry_set.columns:
            industry_components = industry_set[industry_name]
            industry_components = industry_components[factor.index]
            industry_cap[industry_name] = cap_t[industry_components == 1].sum()

        # change the factor loading to satisfy w1 * f1 + w2 * f2 + ... wn * fn
        # = 0, wi, fi are industry cap and industry returns

        for name in industry_key:
            if name != choose:
                factor[name] = factor[name] - industry_cap[name] / \
                    industry_cap[choose] * factor[choose]
        del factor[choose]
        #import pdb; pdb.set_trace()


        model = sm.WLS(returns_t.dropna(), factor.dropna(), weights=cap_t)

        try:
            res = model.fit()
            beta = res.params.copy()
            sum_ret = 0.0

            for name in industry_key:
                if name != choose:
                    sum_ret += industry_cap[name] * beta[name]
            beta[choose] = -1 * sum_ret / industry_cap[choose]

            for s in beta.index:
                if s not in factor_returns.columns:
                    factor_returns[s] = np.nan
            factor_returns.ix[date] = beta

        except Exception as e:
            print e
            factor_returns.ix[date] = 0.0
    factor_returns.index = pd.DatetimeIndex(factor_returns.index)
    factor_returns.to_hdf('E:/multi_factor/factor_returns/factor_returns.h5','table')
