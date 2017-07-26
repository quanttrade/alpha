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
        beta[stock] = model.beta.x
    return beta


def regresi(A, B, n):
    resid = pd.DataFrame(index=A.index, columns=A.columns)
    for stk in A.columns:
        model = pd.stats.ols.MovingOLS(y=B[stk], x=A[stk], window_type='rolling', window=n, intercept=True)
        resid[stock] = model.resid
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
    self.open = pn_data['adjOpen']
    self.high = pn_data['adjHigh']
    self.low = pn_data['adjLow']
    self.close = pn_data['adjClose']
    self.vwap = pn_data['vwap']
    self.volume = pn_data['volume']
    self.returns = self.close.pct_change()
    self.amount = self.volume * self.close


    def alpha001(self):
        return -1 * correlation(rank(delta(log(self.volume), 1)), rank((self.close - self.open) / self.open), 6)


    def alpha002(self):
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) , 1)


    def alpha003(self):
        pass


    def alpha004(self):
        pass


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


    def alpha010(self):
        pass


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
        return rank(self.vwap - ts_max(vwap, 15))**delta(self.close, 5)


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
        self.close / delay(self.close, 6) - 1


    def alpha021(self):
        pass


    def alpha022(self):
        retun sma((self.close - sma(self.close, 6)) / sma(self.close, 6) - delay((self.close - sma(self.close, 6)) / sma(self.close, 6), 3), 12)


    def alpha023(self):
        cond_1 = self.close > delay(self.close, 1)
        cond_2 = self.close <= delay(self.close , 1)
        alpha = stddev(close, 2)
        alpha1 = alpha.copy()
        alpha2 = alpha2.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return sma(alpha2, 20) / (sma(alpha1, 20) + sma(alpha2, 20))

            


    def alpha024(self):
        return sma(self.close - delay(close,5), 5)


    def alpha025(self):
        return -1 * rank(delta(self.close, 7)) * (1 - rank(decay_linear(self.volume / sma(volume, 20), 9)) * (1 + rank(ts_sum(self.returns, 250)))) 


    def alpha026(self):
        return  ts_sum(self.close, 7) / 7 - self.close  + correlation(self.vwap, delay(close, 5), 230)


    def alpha027(self):
        return wma((self.close - delay(self.close, 3)) / delay(self.close, 3) * 100 + (self.close - delay(self.close, 6)) / delay(self.close, 6)* 100, 12)


    def alpha028(self):
        return 3 * sma((self.close - delay(self.close, 3)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3) - 2 * sma(sma((self.close - ts_min(self.low, 9))/ (ts_max(self.high, 9) - ts_max(self.low, 9)) * 100, 3), 3)

        
    def alpha029(self):
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume


    def alpha030(self):
        pass


    def alpha031(self):
         return  (self.close - sma(self.close, 12) / sma(self.close, 12) * 100)


    def alpha032(self):
         -1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)


    def alpha033(self):
        return -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5) * rank(ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) * ts_rank(self.volume, 5)


    def alpha034(self):
        return sma(self.close, 12) / self.close


    def alpha035(self):
        min(rank(decay_linear(delta(self.open,1), 15)), rank(decay_linear(correlation(self.volume, self.open * 0.65 + self.close * 0.35, 17), 7)) * -1)


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
        alpha1 = self.volume
        alpha2 = self.volume
        cond_1 = self.close <= delay(close, 1)
        cond_2 = self > delay(close, 1)
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return ts_sum(alpha1, 26) / ts_sum(alpha2, 26)


    def alpha041(self):
        return rank(ts_max(delta(vwap, 3), 5)) * -1


    def alpha042(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)


    def alpha043(self):
        cond_1 = self.close <= delay(self.close, 1)
        cond_2 = self.close >= delay(self.close)
        alpha1 = self.volume
        alpha2 = -self.volume
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
        bound = max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0
            
        return ts_sum(alpha1, 12) / ts_sum(bound, 12)


    def alpha050(self):
        cond_1 = self.high + self.low <= delay(self.high, 1) + delay(self.low, 1)
        cond_2 = self.high + self.low >= delay(self.high, 1) + delay(self.low, 1)
        bound = max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha2 = bound.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return ts_sum(alpha1, 12) / ts_sum(bound, 12) - ts_sum(alpha2, 12) / ts_sum(bound, 12)


    def alpha051(self):
        cond_1 = self.high + self.low <= delay(self.high, 1) + delay(self.low, 1)
        bound = max(abs(self.high - delay(self.high, 1)), abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0
        return ts_sum(alpha1, 12) / ts_sum(bound, 12)


    def alpha052(self):
        return ts_sum(max(0, self.high - delay((self.high + self.low + self.close) / 3, 1)), 26) / ts_sum(max(0, self.high - delay((self.high + self.low + self.close) / 3, 1) - self.low), 26)


    def alpha053(self):
        return (self.close > delay(self.close, 1)).rolling(12).sum()


    def alpha054(self):
        pass


    def alpha055(self):
        pass


    def alpha056(self):
        pass


    def alpha057(self):
        return sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3)


    def alpha058(self):
        return (self.close > delay(self.close, 1)).rolling(20).sum() / 20 * 100


    def alpha059(self):
        alpha1 = max(self.high, delay(self.close, 1))
        cond_1 = close > delay(self.close, 1)
        alpha1[cond_1] = min(self.low, delay(self.close, 1))
        cond_2 = self.close == delay(self.close, 1)
        alpha2 = alpha1.copy()
        alpha2[cond_2] = 0
        return ts_sum(alpha2, 20)


    def alpha60(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) *self.volume, 20)


    def alpha061(self):
        return 


    def alpha062(self):
        return -1 * correlation(self.high, rank(self.volume), 5)


    def alpha063(self):
        return sma(max(self.close - delay(close, 1), 0), 6) / sma(abs(self.close - delay(self.close, 1)), 6)


    def alpha064(self):
        return max(rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4)), rank(decay_linear(ts_max(correlation(rank(self.close), rank(sma(self.volume, 60)), 4), 13), 14)))


    def alpha065(self):
        return sma(self.close) / self.close


    def alpha066(self):
        return (self.close - sma(self.close, 6)) / sma(self.close, 6) * 100


    def  alpha067(self):
        return sma(max(self.close - delay(self.close, 1), 0), 24) / sma(abs(self.close - delay(self.close, 1)), 24) * 100


    def alpha068(self):
        return sma(((self.high + self.low) / 2 -(delay(self.high, 1) + delay(self.low, 1)) / 2) * (self.high - self.low) / self.volume, 2)


    def alpha069(self):
        pass


    def alpha070(self):
        stddev(self.amount, 6)


    def alpha071(self):
        (self.close - sma(self.close, 24)) / (sma(self.close, 24)) * 100


    def alpha072(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_max(self.low, 6)), 15)


    def alpha073(self):
        return ts_rank(decay_linear(decay_linear(correlation(self.close, self.volume, 10), 16), 4), 5) - ts_rank(decay_linear(correlation(self.vwap, sma(self.volume, 30), 4), 3))






















