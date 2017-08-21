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


    return  pn_data


def fillna_issuingdate(issuingdate):
    pass




def caculate_barra_factor(price_data, fundmental, public_date, benchmark_returns):
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
    public_date.index = pd.DatetimeIndex(public_date.index)
    for descriptor in fundmental_data.keys():
        fundmental_daily = pd.DataFrame(index=price_data['close'][fundmental_data[descriptor].index[0]:].index,
                                        columns=price_data['close'].columns)
        fundmental_quarter = fundmental_data[
            descriptor].copy()[fundmental_daily.columns]

        for stk in fundmental_daily.columns:
            temp = fundmental_quarter[stk]
            to_fill = fundmental_daily[stk].copy()
            for date in temp.index:
                to_fill[public_date[stk].ix[date]:] = temp.ix[date]
            fundmental_daily[stk] = to_fill


        fundmental_daily.index = pd.DatetimeIndex(fundmental_daily.index)


        fundmental_data[descriptor] = fundmental_daily

    # caculate BETA descriptor
    pct_change = price_data['adjclose'].pct_change()
    beta, resid = beta_value(pct_change, benchmark_returns)
    pn_data['BETA'] = beta

    # caculate Momentum descriptor
    Lambda_120 = np.power(0.5, 1.0 / 120.0)
    weight_120 = np.array([Lambda_120 ** (503 - i) for i in range(504)])
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
    weight_40 = np.array([Lambda_40 ** (249 - i) for i in range(250)])
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

    #winsorize and standardize the descriptor
    for descriptor in pn_data.keys():
        pn_data[descriptor] = pn_data[descriptor].ix['2007-03-30':]
        pn_data[descriptor] = standardize(
            winsorize(pn_data[descriptor].copy(), boxplot))



    # caculate Barra risk factor
    barra_factor['Beta'] = pn_data['BETA']
    barra_factor['Momentum'] = pn_data['RSTR']
    barra_factor['Size'] = pn_data['LNCAP']
    barra_factor['NLSIZE'] = pn_data['NLSIZE']
    barra_factor['Earnings Yield'] = descriptor2factor([0.68 * pn_data['EPIBS'],
                                                       0.11 * pn_data['ETOP'], 0.21 * pn_data['CETOP']])
    barra_factor['Volatiliy'] = descriptor2factor(
        [0.74 * pn_data['DASTD'], 0.16 * pn_data['CMRA'], 0.1 * pn_data['HSIGMA']])
    barra_factor['Growth'] = descriptor2factor(
        [0.18 * pn_data['EGIB'], 0.11 * pn_data['EGIB_S'], 0.24 * pn_data['EGRO'], 0.47 * pn_data['SGRO']])
    barra_factor['Value'] = pn_data['BTOP']
    barra_factor['Leverge'] = descriptor2factor(
        [0.38 * pn_data['MLEV'], 0.35 * pn_data['DTOA'], 0.27 * pn_data['BLEV']])
    barra_factor['Liquidity'] = descriptor2factor(
        [0.35 * pn_data['STOM'], 0.35 * pn_data['STOQ'], 0.30 * pn_data['STOA']])



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
    #return the standardized factor
    factor = factor.dropna(how='all')
    factor_std = ((factor.T - factor.mean(axis=1)) / factor.std(axis=1)).T
    return factor_std


def standardize_cap(factor, cap):
    # average factor by cap
    factor_use = factor.copy()
    for date in factor.index:
        factor_t = factor_use.ix[date].dropna()
        cap_t = cap.ix[date].ix[factor_t.index]
        factor_cap = np.average(factor_t, weights=cap_t)
        factor_use.ix[date] = factor_cap
    return factor_use



# caculate beta value
def beta_value(pct_change, benchmark_returns):
    Lambda = np.power(0.5, 1 / 60.0)

    weight = pd.Series([Lambda ** (pct_change.shape[0] - i - 1) for i in range(pct_change.shape[0])],
                       index=pct_change.index)
    pct_change_weight = pct_change.multiply(weight, axis=0)
    benchmark_returns_weight = benchmark_returns.multiply(weight, axis=0)
    beta = pd.DataFrame({},index=pct_change.index, columns=pct_change.columns)
    resid = pd.DataFrame({},index=pct_change.index, columns=pct_change.columns)

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

    return beta,resid


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
        model = sm.OLS(alpha_date, join_factor).fit()

        alpha_neutral.ix[date][alpha_date.index] = model.resid

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



def caculate_factor_returns(barra_factor, industry, price_data):

    for hangye in industry.keys():
        barra_factor[hangye] = industry[hangye]

    volume = price_data['volume']
    returns = price_data['adjclose'].pct_change(1).shift(-1)
    cap = price_data['close'] * price_data['total_shares']

    tradedate = barra_factor.index.get_level_values('tradedate')
    tradedate = list(set(tradedate))
    tradedate.sort()
    #tradedate_M = [tradedate[i] for i in range(1,len(tradedate),21)]

    factor_returns = pd.DataFrame(index=tradedate, columns=barra_factor.columns)
    choose = u'有色金属'

    rsquare = pd.Series(index=tradedate)
    benchmark_returns  = pd.Series(index=tradedate)
    for date in tradedate[:-1]:

        print date

        #choose the stock tradable
        volume_t = volume.ix[date]
        factor = barra_factor.ix[date]
        returns_t = returns.ix[date]
        factor = factor[volume_t > 0]
        factor = factor.replace([0],np.nan)
        factor = factor.dropna(how='all',axis=1)
        factor = factor.fillna(0)

        returns_t = returns_t.ix[factor.index]
        cap_t = cap.ix[date].ix[factor.index]
        industry_key  = factor.columns[10:-1]
        benchmark_return = np.average(returns_t, weights=cap_t)
        benchmark_returns.ix[date] = benchmark_return



        #caculate the cap of every industry
        industry_set = factor[industry_key]
        industry_cap = pd.Series()
        for industry_name in industry_set.columns:
            industry_components = industry_set[industry_name]
            industry_components = industry_components[factor.index]
            industry_cap[industry_name] = cap_t[industry_components == 1].sum()

        #change the factor loading to satisfy w1 * f1 + w2 * f2 + ... wn * fn = 0, wi, fi are industry cap and industry returns

        for name in industry_key:
            if name != choose:
                factor[name] = factor[name] - industry_cap[name] / industry_cap[choose] * factor[choose]
        del factor[choose]


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



if __name__ == "__main__":

    #laoding data

    barra_factor = pd.read_hdf('/Users/liyizheng/data/daily_data/barra_factor.h5','table')
    data = pd.read_hdf('/Users/liyizheng/data/stockdata/data.h5','table')
    prime_close = pd.read_csv('/Users/liyizheng/data/stockdata/prime_close.csv',index_col=0)
    prime_close.index = pd.DatetimeIndex(prime_close.index)
    price_data = load_data(data, prime_close)
