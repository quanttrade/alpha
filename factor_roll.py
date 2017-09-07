import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from alpha_regress import stock_weight
from cvxpy import *


def factor_returns_date(returns, date, path, alpha_list, stop_alpha=None):
    returns_date = returns.ix[date]
    alpha_path = os.path.join(path, 'alpha_factor', '%s.h5' % date)
    barra_path = os.path.join(path, 'barra_factor', '%s.h5' % date)
    alpha_factor_date = pd.read_hdf(alpha_path, 'table').ix[:,:stop_alpha][alpha_list]
    barra_factor_date = pd.read_hdf(barra_path, 'table')
    factor = pd.concat([barra_factor_date, alpha_factor_date], axis=1)
    factor = factor.dropna(how='all', axis=1)
    returns_date = returns_date.ix[factor.index]
    model = sm.OLS(returns_date, factor).fit()
    return model.params


def time_series_return(price, tradedate, path, period, alpha_list, stop_alpha=None):
    factor_returns = pd.DataFrame()
    returns = price.pct_change(period).shift(-period)
    for date in tradedate:
        print date
        date_returns = factor_returns_date(returns, date, path, alpha_list, stop_alpha)
        if len(factor_returns.columns) == 0:
            factor_returns = pd.DataFrame(index=tradedate, columns=date_returns.index)
        for index in date_returns.index:
            if index not in factor_returns.columns:
                factor_returns[index] = np.nan
        factor_returns.ix[date] = date_returns
    return factor_returns


def day_portofolio(price_data, date,  alpha_list, weight_bound, risk_loading_bound, industry_loading_bound, TC, w_old, window, path):

    #loading the data

    cap = price_data['close'] * price_data['total_shares']
    alpha_factor = pd.read_hdf(os.path.join(path, 'alpha_factor', '%s.h5' % date), 'table')[alpha_list]
    barra_factor = pd.read_hdf(os.path.join(path, 'barra_factor', '%s.h5' % date), 'table')
    factor_returns = pd.read_hdf(os.path.join(path, 'factor_returns', 'factor_returns.h5'), 'table').ix[:date].dropna(how='all')
    alpha_returns = factor_returns[alpha_list]
    risk_factor = barra_factor.ix[:,:10]
    industry_factor_t = barra_factor.ix[:,10:-1]
    cap_t = cap.ix[date]
    tradesuspend_stock = w.wset("tradesuspend","startdate=%s;enddate=%s" % (date, date)).Data[1]
    trade_stock = list(set(risk_factor.index) - set(tradesuspend_stock))
    trade_stock.sort()



    benchmark_component_date = pd.read_hdf(os.path.join(path, 'benchmark_component', 'zz500_component.h5'), 'table').ix[date]
    bench_weight = cap_t.ix[risk_factor.index].ix[
        benchmark_component_date].dropna()
    bench_weight = bench_weight / bench_weight.sum()

    bench_risk_loading = risk_factor.ix[
        bench_weight.index].values.T.dot(
        bench_weight.values)
    bench_industry_loading = industry_factor_t.ix[
        bench_weight.index].values.T.dot(bench_weight.values)

    returns = price_data['adjclose'].pct_change().ix[date]

    risk_factor = risk_factor.ix[trade_stock][returns.abs() < 0.098]
    industry_factor_t = industry_factor_t.ix[trade_stock][returns.abs() < 0.098]

    alpha_factor_t = alpha_factor.ix[trade_stock][returns.abs() < 0.098]

    Lambda = np.power(0.5, 1.0 / 60.0)
    decay_weight = np.array([Lambda ** (window - i) for i in range(window)])
    #today's volume and return

    alpha_returns_t = np.average(alpha_returns.ix[
                                    :date][-window :].astype(float), weights=decay_weight, axis=0)

    alpha_returns_t = pd.Series(
        alpha_returns_t, index=alpha_returns.columns)



    # defne the weight vector to solve
    N = risk_factor.shape[0]
    x = Variable(N, 1)
    untrade_stock = []

    # caculate the potofolio loading
    potofolio_risk_loading = risk_factor.values.T * x
    potofolio_industry_loading = industry_factor_t.values.T * x
    alpha_loading = alpha_factor_t.values.T * x
    untrade_weight = 0.0
    # the weight of last day
    if isinstance(w_old, float):
        w_last = 0.0

    else:
        w_last = pd.Series(np.repeat(0.0, N), index=risk_factor.index)
        stock_t = list(set(w_last.index) & set(w_old.index))
        w_last.ix[stock_t] = w_old.ix[stock_t]
        untrade_stock = list(set(w_old.index) - set(w_last.index))
        for stock in untrade_stock:
            untrade_weight += w_old[stock]
        w_last = w_last.values

    # define the object to solve
    ret = alpha_loading.T * alpha_returns_t.values - \
        TC * sum_entries(abs(x - w_last)) / 2.0

    constraints = [
        0 <= x,
        x <= weight_bound,
        sum_entries(x) == 1,
        abs(
            potofolio_risk_loading -
            bench_risk_loading) < risk_loading_bound,
        abs(
            bench_industry_loading -
            potofolio_industry_loading) < industry_loading_bound *
        bench_industry_loading]
    prob = Problem(Maximize(ret), constraints)
    try:
        Maximize_value = prob.solve()
        weight = np.array(x.value)
        weight = pd.Series([weight[i][0] for i in range(
            len(weight))], index=risk_factor.index) * (1 - untrade_weight)
    except Exception as e:
        print e
        weight = w_old
        if weight == 0.0:
            return pd.Series(0,0, index=risk_factor.index)

    if untrade_stock:
        for stock in untrade_stock:
            weight[stock] = w_old[stock]

    weight = weight[weight >= 0.001]
    weight = weight / weight.sum()
    return weight
