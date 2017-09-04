import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from alpha_regress import stock_weight
from cvxpy import *


def factor_returns_date(returns, date, path, stop_alpha=None):
    returns_date = returns.ix[date]
    alpha_path = os.path.join(path, 'alpha_factor', '%s.h5' % date)
    barra_path = os.path.join(path, 'barra_factor', '%s.h5' % date)
    alpha_factor_date = pd.read_hdf(alpha_path, 'table').ix[:,:stop_alpha]
    barra_factor_date = pd.read_hdf(barra_path, 'table')
    factor = pd.concat([barra_factor_date, alpha_factor_date], axis=1)
    factor = factor.dropna(how='all', axis=1)
    returns_date = returns_date.ix[factor.index]
    model = sm.OLS(returns_date, factor).fit()
    return model.params


def time_series_return(price, tradedate, path, period, stop_alpha=None):
    factor_returns = pd.DataFrame()
    returns = price.pct_change(period).shift(-period)
    for date in tradedate:
        print date
        date_returns = factor_returns_date(returns, date, path, stop_alpha)
        if len(factor_returns.columns) == 0:
            factor_returns = pd.DataFrame(index=tradedate, columns=date_returns.index)
        for index in date_returns.index:
            if index not in factor_returns.columns:
                factor_returns[index] = np.nan
        factor_returns.ix[date] = date_returns
    return factor_returns


def day_portofolio(price_data, date,  alpha_list, alpha_returns, weight_bound, risk_loading_bound, industry_loading_bound, TC, w_old, path):

    #loading the data
    cap = price_data['close'] * price_data['total_shares']
    volume = price_data['volume'].copy()
    adjclose = price_data['adjclose'].copy()
    alpha_factor = pd.read_hdf(os.path.join(path, 'alpha_facor', '%s.h5' % date), 'table')[alpha_list]
    barra_factor = pd.read_hdf(os.path.join(path, 'barra_facor', '%s.h5' % date), 'table')
    alpha_returns = pd.read_hdf(os.path.join(path, 'factor_returns', 'factor_returns.h5'), 'table')
    risk_factor = barra_factor.ix[:,:10]
    industry_factor_t = barra_factor.ix[:,10:]


    #today's volume and return
    returns_t = adjclose.pct_change().ix[date]
    volume_t = volume.ix[date]

    benchmark_component_date = pd.read(os.path.join(path, 'benchmark_component', 'zz500_component.h5'), table).ix[date]
    bench_weight = cap_t.ix[risk_factor.index].ix[
        benchmark_component_date].dropna()
    bench_weight = bench_weight / bench_weight.sum()

    return stock_weight(bench_weight, risk_factor, industry_factor_t, volume_t, returns_t, alpha_factor,
                        alpha_returns_t, weight_bound, risk_loading_bound, industry_loading_bound, TC, w_old)
