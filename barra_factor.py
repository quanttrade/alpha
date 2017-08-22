# -*- coding: utf-8 -*-

"""
Created on Wed Mar 29 10:40:11 2017
@author: lh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import medcouple


def load_data(data, prime_close):
    data['tradedate'] = data.tradedate.apply(str)
    data['tradedate'] = pd.DatetimeIndex(data.tradedate)

    prime_close.index = pd.DatetimeIndex(prime_close.index)

    close = data.pivot(index='tradedate',
                       columns='secid',
                       values='closeprice')

    open_ = data.pivot(index='tradedate',
                       columns='secid',
                       values='openprice')

    high = data.pivot(index='tradedate',
                      columns='secid',
                      values='highprice')

    low = data.pivot(index='tradedate',
                     columns='secid',
                     values='lowprice')

    volume = data.pivot(index='tradedate',
                        columns='secid',
                        values='volume')

    total_shares = data.pivot(index='tradedate',
                              columns='secid',
                              values='total_shares')

    vwap = data.pivot(index='tradedate',
                      columns='secid',
                      values='vwap')

    amt = data.pivot(index='tradedate',
                     columns='secid',
                     values='amt')

    free_float_shares = data.pivot(index='tradedate',
                                   columns='secid',
                                   values='free_float_shares')

    adjfactor = prime_close / close

    pn_data = dict()

    pn_data['adjclose'] = close
    pn_data['adjhigh'] = high
    pn_data['adjlow'] = low
    pn_data['adjopen'] = open_
    pn_data['close'] = prime_close
    pn_data['high'] = high * adjfactor
    pn_data['low'] = low * adjfactor
    pn_data['open'] = open_ * adjfactor
    pn_data['vwap'] = vwap * adjfactor
    pn_data['volume'] = volume
    pn_data['total_shares'] = total_shares
    pn_data['free_float_shares'] = free_float_shares
    pn_data['adjvwap'] = vwap
    pn_data['amt'] = amt

    return pn_data


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
    # return the standardized factor
    factor = factor.dropna(how='all')
    factor_std = ((factor.T - factor.mean(axis=1)) / factor.std(axis=1)).T
    return factor_std


def standardize_cap(factor, cap):
    # average factor by cap
    factor_cap = factor.copy()
    for date in factor.index:
        factor_t = factor_cap.ix[date].dropna()
        cap_t = cap.ix[date].ix[factor_t.index]
        factor_cap_average = np.average(factor_t, weights=cap_t)
        factor_cap.ix[date] = (factor_cap.ix[date] - factor_cap_average) / factor_t.std()
    return factor_cap


# caculate beta value
def beta_value(pct_change, benchmark_returns):
    Lambda = np.power(0.5, 1 / 60.0)

    weight = pd.Series([Lambda ** (pct_change.shape[0] - i - 1) for i in range(pct_change.shape[0])],
                       index=pct_change.index)
    pct_change_weight = pct_change.multiply(weight, axis=0)
    benchmark_returns_weight = benchmark_returns.multiply(weight, axis=0)
    beta = pd.DataFrame({}, index=pct_change.index, columns=pct_change.columns)
    resid = pd.DataFrame({}, index=pct_change.index,
                         columns=pct_change.columns)

    for stk in pct_change.columns:
        try:
            ols = pd.stats.ols.MovingOLS(y=pct_change_weight[stk],
                                         x=benchmark_returns_weight,
                                         window=250,
                                         intercept=True)

            beta[stk].ix[ols.beta.x.index] = ols.beta.x
            resid[stk].ix[ols.resid.index] = ols.resid

        except Exception as e:
            print e
            beta[stk] = np.NaN
            beta[stk] = np.NaN

    return beta, resid


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




    alpha_neutral = pd.DataFrame({}, index=alpha.index, columns=alpha.columns)

    for date in alpha.index:
        # togather the data itoday into one dataframe


        factor = factor_list.ix[date]

        factor = factor.replace([0], np.nan)
        factor = factor.dropna(how='all', axis=1)
        factor = factor.fillna(0)


        alpha_date = alpha.ix[date].ix[factor.index]
        alpha_date = alpha_date.fillna(value=alpha_date.quantile())


        # do the ols
        model = sm.OLS(alpha_date, factor.astype(float)).fit()

        alpha_neutral.ix[date].ix[alpha_date.index] = model.resid

    alpha_neutral = alpha_neutral.dropna(how='all', axis=0)
    alpha_neutral = alpha_neutral.dropna(how='all', axis=1)
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


def filter_stock_and_fillna(pn_data, close, N, groupby):
    """
    filter stock in universe more than N days from ipo date

    paramas:
    pn_data: dict
    the dictionary of the descriptor

    close: DataFrame
    the close of each stock

    N: int
    the num we choose to filter

    groupby: dict
    the dict of the Series to classify the stock(the market value, industry etc)
    """

    pct_N = close.pct_change(N)
    filter_universe = pct_N.copy()
    filter_universe = filter_universe.fillna(1)

    # set the value into nan
    for descriptor in pn_data.keys():
        descriptor_df = pn_data[descriptor]
        descriptor_df[filter_universe == 1] = np.NaN
        pn_data[descriptor] = descriptor_df

    for group in groupby.keys():
        group_df = groupby[group]
        group_df[filter_universe == 1] = np.NaN
        groupby[group] = group_df

    df = pd.DataFrame()

    for descriptor in pn_data.keys():
        df[descriptor] = pn_data[descriptor].stack(dropna=False)

    df.index = df.index.rename(['date', 'asset'])

    for group in groupby.keys():
        df[group] = groupby[group].stack(dropna=False)

    grouper = [df.index.get_level_values('date')]
    for group in groupby.keys():
        grouper.append(group)

    grouped = df.groupby(grouper)
    df_fill = grouped.apply(lambda x: x.fillna(x.quantile()))

    return df_fill


def industry_factor(hangye_class):

    hangye_stack = hangye_class.stack()
    hangye = list(set(hangye_stack))
    hangye_dict = dict()

    for industry in hangye:
        temp = pd.Series(0, index=hangye.index)
        temp[hangye == industry] = 1
        hangye_dict[industry] = temp

    return hangye_dict


def caculate_factor_returns(barra_factor,  price_data):

    volume = price_data['volume']
    returns = price_data['adjclose'].pct_change(1).shift(-1)
    cap = price_data['close'] * price_data['total_shares']

    tradedate = barra_factor.index.get_level_values('tradedate')
    tradedate = list(set(tradedate))
    tradedate.sort()
    #tradedate_M = [tradedate[i] for i in range(1,len(tradedate),21)]

    factor_returns = pd.DataFrame(
        index=tradedate, columns=barra_factor.columns)
    choose = u'有色金属'

    rsquare = pd.Series(index=tradedate)

    for date in tradedate[:-1]:

        print date

        # choose the stock tradable
        volume_t = volume.ix[date]
        factor = barra_factor.ix[date]
        returns_t = returns.ix[date]
        factor = factor[volume_t > 0]
        factor = factor.replace([0], np.nan)
        factor = factor.dropna(how='all', axis=1)
        factor = factor.fillna(0)

        returns_t = returns_t.ix[factor.index]
        cap_t = cap.ix[date].ix[factor.index]
        industry_key = factor.columns[10:-1]

        # caculate the cap of every industry
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

        # weighted regression to caculate the returns of each factor
        model = sm.WLS(returns_t.dropna(), factor.dropna(), weights=cap_t)

        try:
            res = model.fit()
            beta = res.params.copy()
            sum_ret = 0.0

            for name in industry_key:
                if name != choose:
                    sum_ret += industry_cap[name] * beta[name]
            beta[choose] = -1 * sum_ret / industry_cap[choose]
            factor_returns.ix[date] = beta
            rsquare.ix[date] = res.rsquared_adj

        except Exception as e:
            print e
            factor_returns.ix[date] = 0.0
    return factor_returns, resid


if __name__ == "__main__":

    # laoding data

    barra_factor = pd.read_hdf(
        '/Users/liyizheng/data/daily_data/barra_factor.h5', 'table')
    data = pd.read_hdf('/Users/liyizheng/data/stockdata/data.h5', 'table')
    prime_close = pd.read_csv(
        '/Users/liyizheng/data/stockdata/prime_close.csv', index_col=0)
    prime_close.index = pd.DatetimeIndex(prime_close.index)
    price_data = load_data(data, prime_close)
