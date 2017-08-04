# -*- coding: utf-8 -*-

"""
Created on Wed Mar 29 10:40:11 2017
@author: lh
"""

import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
import alphalens
from statsmodels import stats
import os

def ts_sum(df, window=10):
    return df.rolling(window).sum()


def sma(df, window=10):
    return df.rolling(window).mean()


def stddev(df, window=10):
    return df.rolling(window).std()


def correlation(x, y, window=10):
    return x.rolling(window).corr(y)


def covariance(x, y, window=10):
    return x.rolling(window).cov(y)


def rolling_rank(na):
    return rankdata(na)[-1]


def ts_rank(x, window=10):
    return x.rolling(window).apply(rolling_rank) / (window + 0.0)


def rolling_prod(na):
    return na.prod(na)


def product(df, window=10):
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    return df.rolling(window).min()


def ts_max(df, window=10):
    return df.rolling(window).max()


def cross_max(df1, df2):
    return (df1 + df2).abs() / 2.0 + (df1 - df2).abs() / 2.0


def cross_min(df1, df2):
    return (df1 + df2).abs() / 2.0 - (df1 - df2).abs() / 2.0


def delta(df, period=1):
    return df.diff(period)


def delay(df, period=1):
    return df.shift(period)


def rank(df):
    return df.rank(axis=1, pct=True) / (df.shape[1] + 0.0)


def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=10):
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=10):
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.ix[:period, :]
    na_series = df.as_matrix()
    divisor = df.as_matrix()
    y = (np.arange(period) + 1) * 1.0 / divisor
    for row in range(period + 1, df.shape[0]):
        x = na_series[row - period + 1:row + 1, :]
        na_lwma[row, :] - (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)


def regbeta(A, B, n):
    beta = pd.DataFrame(index=A.index, columns=A.columns)
    for stk in A.columns:
        model = pd.stats.ols.MovingOLS(y=B[stk], x=A[stk], window_type='rolling', window=n, intercept=True)
        beta[stk] = model.beta.x
    return beta


def regresi(A, B, n):
    resid = pd.DataFrame(index=A.index, columns=A.columns)
    for stk in A.columns:
        model = pd.stats.ols.MovingOLS(y=B[stk], x=A[stk], window_type='rolling', window=n, intercept=True)
        resid[stk] = model.resid
    return resid


def wma(A, n):
    weight = np.array([0.9**i for i in range(n)])
    return A.rolling(n).apply(lambda x: x.T.dot(weight))




class GtjaAlpha(object):

    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.open = pn_data['open']
        self.high = pn_data['high']
        self.low = pn_data['low']
        self.close = pn_data['close']
        self.vwap = pn_data['vwap']
        self.volume = pn_data['volume']
        self.returns = self.close.pct_change()
        self.amount = self.volume * self.close


    def alpha001(self):
        return -1 * correlation(rank(delta(log(self.volume), 1)), rank((self.close - self.open) / self.open), 6)


    def alpha002(self):
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) , 1)


    def alpha003(self):
        cond_1 = self.close == delay(self.close, 1)
        cond_2 = self.close > delay(self.close, 1)
        alpha = self.close - cross_max(self.high, delay(self.close, 1))
        alpha[cond_2] = self.close - cross_min(self.low, delay(self.close, 1))
        alpha[cond_1] = 0
        return ts_sum(alpha, 6)




    def alpha005(self):
        return  -1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)


    def alpha006(self):
        return -1 * (rank(sign(delta((self.open * 0.85) + (self.high * 0.15), 4))))


    def alpha007(self):
        return rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3)) * rank(delta(self.volume, 3))


    def alpha008(self):
        return rank(delta((self.high + self.low) / 2 * 0.2 + self.vwap * 0.8, 4) * -1)


    def alpha009(self):
        return sma(((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2) * ((self.high - self .low) / self.volume), 7)


    def alpha011(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 6 )


    def alpha012(self):
        return rank(self.open - ts_sum(self.vwap, 10) / 10) * (-1 * (rank(abs(self.close - self.vwap))))


    def alpha013(self):
        return (self.high * self.low)**0.5 / self.vwap


    def alpha014(self):
        return self.close / delay(self.close, 5)


    def alpha015(self):
        return self.open / delay(self.close, 1) - 1


    def alpha016(self):
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha017(self):
        return rank(self.vwap - ts_max(self.vwap, 15))**delta(self.close, 5)


    def alpha018(self):
        return self.close / delay(self.close, 5)


    def alpha019(self):
        cond_1 = self.close < delay(self.close, 5)
        cond_2 = self.close == delay(self.close, 5)
        alpha = (self.close - delay(self.close, 5)) / self.close
        alpha[cond_2] = 0
        alpha[cond_1] = (self.close - delay(self.close, 5)) / delay(self.close, 5)
        return alpha


    def alpha020(self):
        return self.close / delay(self.close, 6) - 1



    def alpha022(self):
        return sma((self.close - sma(self.close, 6)) / sma(self.close, 6) - delay((self.close - sma(self.close, 6)) / sma(self.close, 6), 3), 12)


    def alpha023(self):
        cond_1 = self.close > delay(self.close, 1)
        cond_2 = self.close <= delay(self.close , 1)
        alpha = stddev(self.close, 20)
        alpha1 = alpha.copy()
        alpha2 = alpha.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return sma(alpha2, 20) / (sma(alpha1, 20) + sma(alpha2, 20))

            
    def alpha024(self):
        return sma(self.close - delay(self.close,5), 5)


    def alpha025(self):
        return -1 * rank(delta(self.close, 7)) * (1 - rank(decay_linear(self.volume / sma(self.volume, 20), 9)) * (1 + rank(ts_sum(self.returns, 250)))) 


    def alpha026(self):
        return  ts_sum(self.close, 7) / 7 - self.close  + correlation(self.vwap, delay(self.close, 5), 230)


    def alpha027(self):
        return wma((self.close - delay(self.close, 3)) / delay(self.close, 3) * 100 + (self.close - delay(self.close, 6)) / delay(self.close, 6)* 100, 12)


    def alpha028(self):
        return 3 * sma((self.close - delay(self.close, 3)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3) - 2 * sma(sma((self.close - ts_min(self.low, 9))/ (ts_max(self.high, 9) - ts_max(self.low, 9)) * 100, 3), 3)

        
    def alpha029(self):
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume



    def alpha031(self):
         return  (self.close - sma(self.close, 12) / sma(self.close, 12) * 100)


    def alpha032(self):
         return -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)


    def alpha033(self):
        return -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5) * rank(ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) * ts_rank(self.volume, 5)


    def alpha034(self):
        return sma(self.close, 12) / self.close


    def alpha035(self):
        return cross_min(rank(decay_linear(delta(self.open,1), 15)), rank(decay_linear(correlation(self.volume, self.open * 0.65 + self.close * 0.35, 17), 7)) * -1)


    def alpha036(self):
        return rank(ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2))


    def alpha037(self):
        return -1 * rank(ts_sum(self.open, 5) * ts_sum(self.returns, 5) - delay(ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)


    def alpha038(self):
        cond = ts_sum(self.high, 20) / 20 -self.high >= 0
        alpha = -1 * delta(self.high, 2)
        alpha[cond] = 0
        return alpha


    def alpha039(self):
         return -rank(decay_linear(delta(self.close, 2), 8)) + rank(decay_linear(correlation(0.3 * self.vwap + 0.7 * self.open, ts_sum(sma(self.volume, 180), 37), 14), 12)) 


    def alpha040(self):
        alpha1 = self.volume.copy()
        alpha2 = self.volume.copy()
        cond_1 = self.close <= delay(self.close, 1)
        cond_2 = self > delay(self.close, 1)
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return ts_sum(alpha1, 26) / ts_sum(alpha2, 26)


    def alpha041(self):
        return rank(ts_max(delta(self.vwap, 3), 5)) * -1


    def alpha042(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)


    def alpha043(self):
        cond_1 = self.close <= delay(self.close, 1)
        cond_2 = self.close >= delay(self.close)
        alpha1 = self.volume.copy()
        alpha2 = -self.volume.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        alpha = alpha1 + alpha2
        return ts_sum(alpha, 2)


    def alpha044(self):
        return ts_rank(decay_linear(correlation(self.low, sma(self.volume, 10), 7), 6), 4) + ts_rank(decay_linear(delta(self.vwap, 3), 10), 15)


    def alpha045(self):
        return rank(delta(self.close * 0.6 + self.open * 0.4, 1)) * rank(correlation(self.vwap, sma(self.volume, 150), 15))


    def alpha046(self):
        return (sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close , 24)) / (self.close * 4)


    def alpha047(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9)


    def alpha048(self):
        return -1 * rank(sign(self.close - delay(self.close, 1)) + sign(delay(self.close, 1) - delay(self.close, 2)) + sign(delay(self.close, 2) - delay(self.close, 3))) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)


    def alpha049(self):
        cond_1 = self.high + self.low >= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0
            
        return ts_sum(alpha1, 12) / ts_sum(bound, 12)


    def alpha050(self):
        cond_1 = self.high + self.low <= delay(self.high, 1) + delay(self.low, 1)
        cond_2 = self.high + self.low >= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha2 = bound.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return ts_sum(alpha1, 12) / ts_sum(bound, 12) - ts_sum(alpha2, 12) / ts_sum(bound, 12)


    def alpha051(self):
        cond_1 = self.high + self.low <= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0
        return ts_sum(alpha1, 12) / ts_sum(bound, 12)


    def alpha052(self):
        return ts_sum(cross_max(0, self.high - delay((self.high + self.low + self.close) / 3, 1)), 26) / ts_sum(cross_max(0, self.high - delay((self.high + self.low + self.close) / 3, 1) - self.low), 26)


    def alpha053(self):
        return (self.close > delay(self.close, 1)).rolling(12).sum()



    def alpha057(self):
        return sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3)


    def alpha058(self):
        return (self.close > delay(self.close, 1)).rolling(20).sum() / 20 * 100


    def alpha059(self):
        alpha1 = cross_max(self.high, delay(self.close, 1))
        cond_1 = self.close > delay(self.close, 1)
        alpha1[cond_1] = cross_min(self.low, delay(self.close, 1))
        cond_2 = self.close == delay(self.close, 1)
        alpha2 = alpha1.copy()
        alpha2[cond_2] = 0
        return ts_sum(alpha2, 20)


    def alpha60(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) *self.volume, 20)



    def alpha062(self):
        return -1 * correlation(self.high, rank(self.volume), 5)


    def alpha063(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 6) / sma(abs(self.close - delay(self.close, 1)), 6)


    def alpha064(self):
        return cross_max(rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4)), rank(decay_linear(ts_max(correlation(rank(self.close), rank(sma(self.volume, 60)), 4), 13), 14)))


    def alpha065(self):
        return sma(self.close) / self.close


    def alpha066(self):
        return (self.close - sma(self.close, 6)) / sma(self.close, 6) * 100


    def  alpha067(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 24) / sma(abs(self.close - delay(self.close, 1)), 24) * 100


    def alpha068(self):
        return sma(((self.high + self.low) / 2 -(delay(self.high, 1) + delay(self.low, 1)) / 2) * (self.high - self.low) / self.volume, 2)



    def alpha070(self):
        return stddev(self.amount.pct_change(), 6)


    def alpha071(self):
        return (self.close - sma(self.close, 24)) / (sma(self.close, 24)) * 100


    def alpha072(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_max(self.low, 6)), 15)


    def alpha073(self):
        return ts_rank(decay_linear(decay_linear(correlation(self.close, self.volume, 10), 16), 4), 5) - rank(decay_linear(correlation(self.vwap, sma(self.volume, 30), 4), 3))


    def alpha074(self):
        return rank(correlation(ts_sum(self.low * 0.35 + self.vwap * 0.65, 20), ts_sum(sma(self.volume, 40), 20), 7)) + rank(correlation(rank(self.vwap), rank(self.volume), 6))
    
    
    def alpha076(self):
        return stddev((self.close / delay(self.close) - 1).abs() / self.volume, 20) / sma((self.close / delay(self.close) - 1).abs() / self.volume, 20)
    
    
    def alpha077(self):
        return cross_min(rank(decay_linear(self.high / 2.0 + self.low / 2.0 - self.vwap), 20), rank(decay_linear(correlation((self.high + self.low) / 2.0, sma(self.volume, 40), 3), 6)))
    
    
    def alpha078(self):
        return ((self.high + self.low + self.close) / 3.0 - sma((self.high + self.low + self.close) / 3.0, 12)) / sma((self.close - sma((self.high + self.low + self.close) / 3.0, 12)).abs(), 12)
    
    
    def alpha079(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 12) / sma((self.close - delay(self.close, 1)).abs(), 12)
    
    
    def alpha080(self):
        return (self.volume - delay(self.volume, 5)) / delay(self.volume, 5)
    
    
    def alpha081(self):
        return sma(self.volume, 21) / self.volume
    
    
    def alpha082(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)), 20)
    
    
    def alpha083(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))
    
    
    def alpha085(self):
        return ts_rank(self.volume / sma(self.volume, 20.0), 20) * ts_rank(-1 * delta(self.close,7) / self.close, 8)
    
    
    def alpha087(self):
        return -1 * rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(self.low - self.vwap) / (self.open - (self.low + self.high) / 2.0, 11), 7)
    
    def alpha088(self):
        return self.close / delay(self.close, 20) -1
    
    
    def alpha090(self):
        return -1 *rank(correlation(rank(self.vwap), rank(self.volume), 5))
        
    def alpha091(self):
        return -1 * rank(self.close / ts_max(self.close, 5)) * rank(correlation(sma(self.volume, 40), self.low, 5))
    
    def alpha093(self):
        cond = self.open > delay(self.open, 1)
        alpha = cross_max(self.open - self.low, self.open - delay(self.open, 1))
        alpha[cond] = 0
        alpha = alpha / self.close
        return ts_sum(alpha, 20)
    
    
    def alpha095(self):
        return correlation(self.high / self.low, self.volume, 6).replace([np.inf, -np.inf], np.nan)
    
    
    
    def alpha096(self):
        return sma(sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)), 3), 3)






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


def format_factor(factor):
    """
    params: factor formating like DataFrame
    returns: factor formating like Series
    """

    it = [list(pd.DatetimeIndex(factor.index)), list(factor.columns)]
    index = pd.MultiIndex.from_product(it, names=['tradeDate', 'secID'])
    factor_data = pd.DataFrame(factor.values.reshape(-1, 1), index=index)[0]
    return factor_data.dropna()



def winsorize_series(se):
    median = se.quantile(0.5)
    mad = np.abs(se - median).quantile(0.5)
    se[se < (median - 3 * 1.4286 * mad)] = median - 3 * 1.4286 * mad
    se[se > (median + 3 * 1.4286 * mad)] = median + 3 * 1.4286 * mad
    return se


def winsorize(factor):
    return factor.apply(winsorize_series, axis=1)


def standardize(factor):
    factor = factor.dropna(how='all')
    factor_std = ((factor.T - factor.mean(axis=1)) / factor.std(axis=1)).T
    return factor_std


def handle_factor(factor, prices, groupby, periods, path):
    factor_format = format_factor(factor)
    prices_format = prices.ix[factor_format.index[0][0]:]



    # standard factor performance

    factor_data_standard = alphalens.utils.get_clean_factor_and_forward_returns(factor_format,
                                                                                prices_format,
                                                                                periods=periods)

    quantile_returns_mean_standard, quantile_returns_std_standard = alphalens.performance.mean_return_by_quantile(
        factor_data_standard)

    ic_standard = alphalens.performance.factor_information_coefficient(
        factor_data_standard)

    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_standard.mean()
    ic_summary_table["IC Std."] = ic_standard.std()
    t_stat, p_value = stats.ttest_1samp(ic_standard, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_standard)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_standard)
    ic_summary_table["Ann. IR"] = (
        ic_standard.mean() / ic_standard.std()) * np.sqrt(252)


    
    factor.to_excel(path + '\\prime_factor.xlsx')
    factor_data_standard.to_excel(path + '\\factor_data_standard.xlsx')
    quantile_returns_mean_standard.to_excel(path + '\\quantile_returns_mean_standard.xlsx')
    ic_standard.to_excel(path + '\\ic_standard.xlsx')
    ic_summary_table.to_excel(path + '\\ic_summary_table.xlsx')




    # factor performance in different group(by industry, size, value ...etc)

    def information_statistcs(ic):              #caculate ic_statitic use the daily ic data
        ic_group = ic.groupby(ic.index.get_level_values('group'))
        ic_mean = ic_group.agg(np.mean)
        ic_std = ic_group.agg(np.std)
        ic_tsta = ic_group.agg(lambda x: (stats.ttest_1samp(x, 0)[0]))
        ic_pvalue = ic_group.agg(lambda x: (stats.ttest_1samp(x, 0)[1]))
        ic_skew = ic_group.agg(stats.skew)
        ic_kurtosis = ic_group.agg(stats.kurtosis)
        ic_ir = ic_mean / ic_std * np.sqrt(252)

        ic_mean.index = map(lambda x: x + '_ic_mean', ic_mean.index)
        ic_std.index = map(lambda x: x + '_ic_std', ic_std.index)
        ic_tsta.index = map(lambda x: x + '_ic_tsta', ic_tsta.index)
        ic_pvalue.index = map(lambda x: x + '_ic_pvalue', ic_pvalue.index)
        ic_skew.index = map(lambda x: x + '_ic_skew', ic_skew.index)
        ic_kurtosis.index = map(
            lambda x: x + '_ic_kurtosis', ic_kurtosis.index)
        ic_ir.index = map(lambda x: x + '_ic_ir', ic_ir.index)

        return pd.concat([ic_mean, ic_std, ic_tsta, ic_pvalue, ic_skew, ic_kurtosis, ic_ir])

    for key in groupby.keys():

        #generate the factor_data with forward returns and group

        factor_data_key = alphalens.utils.get_clean_factor_and_forward_returns(factor_format,
                                                                               prices_format,
                                                                               periods=periods,
                                                                               groupby=groupby[
                                                                                   key],
                                                                               by_group=True)

        #caculate factor returns with group_neutral

        factor_returns = alphalens.performance.factor_returns(
            factor_data_key, group_neutral=True)

        #caculate quantile returns_mean

        quantile_returns_mean_key, quantile_returns_std_key = alphalens.performance.mean_return_by_quantile(
            factor_data_key, by_group=True)

        # caculate ic by group

        ic = alphalens.performance.factor_information_coefficient(
            factor_data_key, by_group=True, group_adjust=True)

        ic_table = information_statistcs(ic)

        # turnover data

        turnover_periods = alphalens.utils.get_forward_returns_columns(
            factor_data_key.columns)
        quantile_factor = factor_data_key['factor_quantile']

        quantile_turnover = {p: pd.concat([alphalens.performance.quantile_turnover(
            quantile_factor, q, p) for q in range(1, int(quantile_factor.max()) + 1)],
            axis=1)
            for p in turnover_periods}


        quantile_turnover_mean = pd.Panel(quantile_turnover).mean()


        #save data to excel
        factor_data_key.to_excel(path + '\\factor_data_%s.xlsx' % key)
        factor_returns.to_excel(path + '\\factor_returns_%s.xlsx' % key)
        quantile_returns_mean_key.to_excel(path + '\\quantile_returns_mean_%s.xlsx' % key)
        ic.to_excel(path + '\\ic_%s.xlsx' % key)
        ic_table.to_excel(path + '\\ic_table_%s.xlsx' % key)
        quantile_turnover_mean.to_excel(path + '\\quantile_turnover_mean_%s.xlsx' % key)


if __name__ == "__main__":

    # get alpha function of GtjaAlpha

    alpha_function = GtjaAlpha.__dict__.keys()
    alpha_function.sort()
    alpha_function = alpha_function[5:]

    # load price and volume data

    data = pd.read_csv('/Users/liyizheng/data/daily_data/stock_data.csv')
    data, pn_data = load_data(data)
    gtja = GtjaAlpha(pn_data)

    # get class signal by hs300 and zz500
    hs300 = pd.read_csv(
        '/Users/liyizheng/data/daily_data//hs300_component.csv', index_col=0)
    zz500 = pd.read_csv(
        '/Users/liyizheng/data/daily_data//zz500_component.csv', index_col=0)
    stock = [list(pn_data['volume'].columns) for i in range(hs300.shape[0])]
    stock = pd.DataFrame(stock, index=hs300.index)
    signal = pd.DataFrame(u'其余股票', index=stock.index,
                          columns=pn_data['volume'].columns)
    for date in stock.index:
        hs300_td = hs300.ix[date]
        zz500_td = zz500.ix[date]
        stock_td = stock.ix[date]
        signal.ix[date][set(hs300_td) & set(signal.columns)] = u'沪深300'
        signal.ix[date][set(zz500_td) & set(signal.columns)] = u'中证500'

    signal = format_factor(signal)

    groupby = dict()
    groupby['cap'] = signal

    # set the period
    periods = [1, 2, 4, 5, 10, 20]

    path = ''

    # caculate alpha_data and analyse
    for alpha_name in alpha_function:
        print "========================caculating %s =============================" % alpha_name
        try:
            alpha = eval('gtja.%s()' % alpha_name)
            
            # cleaning and standardize alpha data
            alpha = alpha.replace([-np.inf, np.inf], np.nan)
            alpha = winsorize(alpha)
            alpha = standardize(alpha)
            
            
            if not os.path.exists(path + '\%s' % alpha_name):
                os.mkdir(path + '\%s' % alpha_name)

            handle_factor(alpha, gtja.close.copy(), groupby, periods, path + '\%s' % alpha_name)
            
            del alpha

        except Exception as e:
            print e

