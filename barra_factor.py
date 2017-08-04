# -*- coding: utf-8 -*-

"""
Created on Wed Mar 29 10:40:11 2017
@author: lh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import medcouple


def caculate_barra_factor(price_data, fundmental, benchmark_returns):
    """
    return the barra factor based on USE4 and CNE5

    paramas: 
    ----------------------------
    price_data:  dict
    the price and volume based daily data of stock

    fundmental: DataFrame
    the fundmental data of stock , update per quarter

    benchmark_returns: Series
    the returns series of benchmark

    """

    # define fundmental data and descriptor data dict
    fundmental_data = dict()
    pn_data = dict()
    barra_factor = dict()

    # release data from fundmental
    for descriptor in fundmental.columns:
        if descriptor not in ['TRADE_DATE', 'SECID']:
            fundmental_data[descriptor] = fundmental.pivot(
                index='TRADE_DATE', columns='SECID', values=descriptor)

    # transfer quarter fundmental data into daily form
    for descriptor in fundmental_data.keys():
        fundmental_daily = pd.DataFrame(index=price_data['close'][fundmental_data[descriptor].index[0]:].index,
                                        columns=price_data['close'].columns)
        fundmental_quarter = fundmental_data[
            descriptor].copy()[fundmental_daily.columns]

        fundmental_daily.index = pd.DatetimeIndex(fundmental_daily.index)

        fundmental_daily.ix[fundmental_quarter.index[:-1]] = fundmental_quarter
        fundmental_daily = fundmental_daily.fillna(method='pad', limit=63)
        fundmental_data[descriptor] = fundmental_daily

    # caculate BETA descriptor
    pct_change = price_data['adjclose'].pct_change()
    beta, resid = beta_value(pct_change.fillna(0), benchmark_returns)
    pn_data['BETA'] = beta

    # caculate Momentum descriptor
    Lambda_120 = np.power(0.5, 1.0 / 120.0)
    weight_120 = np.array([Lambda_120 ** i for i in range(504)])
    momentum = pct_change.rolling(504 + 21).apply(lambda x: np.average(
        np.log(1 + x[21:]), weights=weight_120, axis=0))
    pn_data['RSTR'] = momentum

    # caculate SIZE descriptor
    fundmental_data['MKT_CAP_ARD'] = price_data[
        'close'] * price_data['total_shares']
    pn_data['LNCAP'] = np.log(fundmental_data['MKT_CAP_ARD'])

    # caculate NLSIZE descriptor
    standardize_cap = standardize(winsorize(pn_data['LNCAP'].copy(), boxplot))
    cap_cube = standardize_cap ** 3
    count = standardize_cap.count(axis=1)
    b = (count * (standardize_cap * cap_cube).sum(axis=1) - standardize_cap.sum(axis=1) *
         cap_cube.sum(axis=1)) / (count * (standardize_cap ** 2).sum(axis=1) - (standardize_cap.sum(axis=1))**2)
    pn_data['NLSIZE'] = cap_cube - standardize_cap.multiply(b, axis=0)

    # caculate Earnings Yield descriptor
    pn_data['EPIBS'] = fundmental_data[
        'WEST_NETPROFIT_FTM'] / fundmental_data['MKT_CAP_ARD']
    pn_data['ETOP'] = fundmental_data[
        'PROFIT_TTM'] / fundmental_data['MKT_CAP_ARD']
    pn_data['CETOP'] = fundmental_data[
        'OPERATECASHFLOW_TTM'] / fundmental_data['MKT_CAP_ARD']

    # caculate Volatility descriptor
    Lambda_40 = np.power(0.5, 1 / 40.0)
    weight_40 = np.array([Lambda_40 ** i for i in range(250)])
    pct_change = price_data['adjclose'].pct_change()
    pn_data['DASTD'] = pct_change.rolling(250).apply(
        lambda x: np.average((x - x.mean(axis=0)) ** 2, weights=weight_40, axis=0))

    pct_21 = price_data['adjclose'].pct_change(21)
    pn_data['CMRA'] = np.log(pct_21.rolling(
        252).max() + 1) - np.log(pct_21.rolling(252).min() + 1)

    Lambda_60 = np.power(0.5, 1 / 60.0)
    weight = pd.Series([Lambda_60 ** (pct_change.shape[0] - i - 1) for i in range(pct_change.shape[0])],
                       index=pct_change.index)
    hsigma = resid.divide(weight, axis=0).rolling(250).std()
    pn_data['HSIGMA'] = hsigma

    # caculate Growth descriptor
    pn_data['SGRO'] = fundmental_data['GROWTH_GR']
    pn_data['EGRO'] = fundmental_data['GROWTH_NETPROFIT']
    pn_data['EGIB'] = fundmental_data['WEST_NETPROFIT_CAGR']
    pn_data['EGIB_S'] = fundmental_data['WEST_AVGNP_YOY']

    # caculate Value descriptor
    pn_data['BTOP'] = fundmental_data[
        'EQUITY_MRQ'] / fundmental_data['MKT_CAP_ARD']

    # caculate Leverge descriptor
    pn_data['MLEV'] = (fundmental_data['MKT_CAP_ARD'] +
                       fundmental_data['WGSD_DEBT_LT']) / fundmental_data['MKT_CAP_ARD']
    pn_data['DTOA'] = fundmental_data[
        'WGSD_LIABS'] / fundmental_data['WGSD_ASSETS']
    pn_data['BLEV'] = (fundmental_data['EQUITY_MRQ'] +
                       fundmental_data['WGSD_DEBT_LT']) / fundmental_data['EQUITY_MRQ']

    # caculate Liquidity descriptor
    stom = np.log(
        (price_data['volume'] / price_data['free_float_shares']).rolling(21).sum())
    pn_data['STOM'] = stom
    pn_data['STOQ'] = np.log(np.exp(stom).rolling(3).mean())
    pn_data['STOA'] = np.log(np.exp(stom).rolling(12).mean())

    for descriptor in pn_data.keys():
        pn_data[descriptor] = pn_data[descriptor].ix['2007-03-30':]
        pn_data[descriptor] = standardize(
            winsorize(pn_data[descriptor], boxplot))

    # caculate Barra risk factor
    barra_factor['Beta'] = pn_data['BETA']
    barra_factor['Momentum'] = pn_data['RSTR']
    barra_factor['Size'] = pn_data['LNCAP']
    barra_factor['NLSIZE'] = pn_data['NLSIZE']
    barra_factor['Earnings Yield'] = descriptor2factor(0.68 * pn_data['EPIBS'],
                                                       0.11 * pn_data['ETOP'], 0.21 * pn_data['CETOP'])
    barra_factor['Volatiliy'] = descriptor2factor(
        0.74 * pn_data['DASTD'], 0.16 * pn_data['CMRA'], 0.1 * pn_data['HSIGMA'])
    barra_factor['Growth'] = descriptor2factor(
        0.18 * pn_data['EGIB'], 0.11 * pn_data['EGIB_S'], 0.24 * pn_data['EGRO'], 0.47 * pn_data['SGRO'])
    barra_factor['Value'] = pn_data['BTOP']
    barra_factor['Leverge'] = descriptor2factor(
        0.38 * pn_data['MLEV'], 0.35 * pn_data['DTOA'], 0.27 * pn_data['BLEV'])
    barra_factor['Liquidity'] = descriptor2factor(
        0.35 * pn_data['STOM'], 0.35 * pn_data['STOQ'], 0.30 * pn_data['STOA'])

    # filling missing value of barra factor


#winsorize and standardize
def mad_method(se):
    median = se.quantile(0.5)
    mad = np.abs(se - median).quantile(0.5)
    se[se < (median - 5.5 * mad)] = median - 5.5 * mad
    se[se > (median + 5.5 * mad)] = median + 5.5 * mad
    return se


def boxplot(data):
    # mc可以使用statsmodels包中的medcouple函数直接进行计算
    mc = medcouple(data.dropna())
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    if mc >= 0:
        l = q1 - 1.5 * np.exp(-3.5 * mc) * iqr
        u = q3 + 1.5 * np.exp(4 * mc) * iqr
    else:
        l = q1 - 1.5 * np.exp(-4 * mc) * iqr
        u = q3 + 1.5 * np.exp(3.5 * mc) * iqr
    data = pd.Series(data)
    data[data < l] = l
    data[data > u] = u
    return data


def winsorize(factor, winsorize_series):
    factor = factor.dropna(how='all')
    return factor.apply(winsorize_series, axis=1)


def standardize(factor):
    factor = factor.dropna(how='all')
    factor_std = ((factor.T - factor.mean(axis=1)) / factor.std(axis=1)).T
    return factor_std


# caculate beta value
def beta_value(pct_change, benchmark_returns):
    Lambda = np.power(0.5, 1 / 60.0)

    weight = pd.Series([Lambda ** (pct_change.shape[0] - i - 1) for i in range(pct_change.shape[0])],
                       index=pct_change.index)
    pct_change_weight = pct_change.multiply(weight, axis=0)
    benchmark_returns_weight = benchmark_returns.multiply(weight, axis=0)
    beta = dict()
    resid = dict()

    for stk in pct_change.columns:
        ols = pd.stats.ols.MovingOLS(y=pct_change_weight[stk],
                                     x=benchmark_returns_weight,
                                     window=250,
                                     intercept=True)

        beta[stk] = ols.beta.x
        resid[stk] = ols.resid

    return pd.DataFrame(beta), pd.DataFrame(resid)


def descriptor2factor(descriptor_list):
    return reduce(lambda x, y: x.add(y, fill_value=0), descriptor_list)


class NonMatchingTimezoneError(Exception):
    pass


def neutralize(alpha, factor_list):
    """
    returns the neutralized result of alpha

    params:
    =======
    alpha : DataFrame
    the factor to be neutralized

    factor_list: list
    the factor used to regress alpha
    """
    for factor in factor_list:
        diff = set(alpha.index) - set(factor.index)
        if len(diff) > 0:
            raise NonMatchingTimezoneError(
                "The timezone of '%s' is not the same as the timezone of 'alpha'." % factor)

    alpha_neutral = pd.DataFrame({}, index=alpha.index, columns=alpha.columns)

    for date in alpha.index:

        # togather the data itoday into one dataframe
        join_factor = reduce(lambda x, y: pd.concat([x, y]), [
                             factor.ix[date].dropna() for factor in factor_list])
        alpha_date = alpha.ix[date].dropna()

        # do the ols
        beta = np.matrix(join_factor.dot(join_factor.T)).I.dot(
            alpha_date.dot(join_factor))

        if len(join_factor.shape) > 1:
            resid = alpha_date - np.matrix(join_factor.T).dot(beta)
        else:
            resid = alpha_date - join_factor * beta.tolist()[0]

        resid = resid.apply(float)

        alpha_neutral.ix[date] = resid

    return alpha_neutral


def fillna_quantile(factor_list, industry_class, cap):
    factor_fills = []
    for factor in factor_list:
        # convert dataframe to stacked forms
        factor_stack = factor.stack(dropna=False)
        factor_stack.to_frame(name='factor')
        factor_stack.index = factor_stack.index.rename(['date', 'asset'])

        # insert grouper columns into stack
        factor_stack['industry'] = industry_class[factor_stack.index[0][0]:]
        factor_stack['cap'] = cap[factor_stack[0][0]:]

        # group factor by industry and cap
        grouper = [factor_stack.index.get_level_values('date')]
        grouper.append('industry')
        grouper.append('cap')

        # fillna with quantile in group
        grouped = factor_stack.groupby(grouper)
        factor_fill = grouped.apply(lambda x: x.fillna(x.quantile()))
        factor_fills.append(factor_fill)

    return factor_fills
