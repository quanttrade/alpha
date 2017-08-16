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


def cross_max(df1, df2):
    return (df1 + df2) / 2.0 + (df1 - df2).abs() / 2.0


def cross_min(df1, df2):
    return (df1 + df2) / 2.0 - (df1 - df2).abs() / 2.0


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
    weight = period - np.arange(period)
    return df.rolling(period).apply(lambda x: np.average(x, weights=weight))


def regbeta(A, B, n):
    beta = pd.DataFrame(index=A.index, columns=A.columns)
    if isinstance(B, pd.DataFrame) and A.shape == B.shape:
        for stk in A.columns:
            model = pd.stats.ols.MovingOLS(
                y=A[stk], x=B[stk], window_type='rolling', window=n, intercept=True)
        beta[stk] = model.beta.x

    if isinstance(B, pd.DataFrame) and B.shape[0] == n:
        for stk in A.columns:
            x = B[stk]
            beta[stk] = A[stk].rolling(n).apply(lambda y: (
                n * x.dot(y) - x.sum() * y.sum() / (n * (x ** 2).sum() - (x.sum()) ** 2)))

    if isinstance(B, pd.Series) and len(B) == n:
        for stk in A.columns:
            x = B
            beta[stk] = A[stk].rolling(n).apply(lambda y: (
                n * x.dot(y) - x.sum() * y.sum() / (n * (x ** 2).sum() - (x.sum()) ** 2)))

    return beta


def regresi(A, B, n):
    resid = pd.DataFrame(index=A.index, columns=A.columns)
    for stk in A.columns:
        model = pd.stats.ols.MovingOLS(
            y=B[stk], x=A[stk], window_type='rolling', window=n, intercept=True)
        resid[stk] = model.resid
    return resid


def wma(A, n):
    weight = np.array([0.9**i for i in range(n)])
    return A.rolling(n).apply(lambda x: x.T.dot(weight))


def highday(df, window=10):
    return (ts_argmax(df, window) - window).abs()


def lowday(df, window=10):
    return (ts_argmin(df, window) - window).abs()


def count(condition, n):
    return condition.rolling(n).sum()


def sumif(df, n, condition):
    alpha = df.copy()
    alpha[True - condition] = 0
    return ts_sum(alpha, n)


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
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low), 1)

    def alpha003(self):
        cond_1 = self.close == delay(self.close, 1)
        cond_2 = self.close > delay(self.close, 1)
        alpha = self.close - cross_max(self.high, delay(self.close, 1))
        alpha[cond_2] = self.close - cross_min(self.low, delay(self.close, 1))
        alpha[cond_1] = 0
        return ts_sum(alpha, 6)

    def alpha005(self):
        return -1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    def alpha006(self):
        return -1 * (rank(sign(delta((self.open * 0.85) + (self.high * 0.15), 4))))

    def alpha007(self):
        return rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3)) * rank(delta(self.volume, 3))

    def alpha008(self):
        return rank(delta((self.high + self.low) / 2 * 0.2 + self.vwap * 0.8, 4) * -1)

    def alpha009(self):
        return sma(((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2) * ((self.high - self .low) / self.volume), 7)

    def alpha011(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 6)

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
        alpha[cond_1] = (self.close - delay(self.close, 5)) / \
            delay(self.close, 5)
        return alpha

    def alpha020(self):
        return self.close / delay(self.close, 6) - 1

    def alpha022(self):
        return sma((self.close - sma(self.close, 6)) / sma(self.close, 6) - delay((self.close - sma(self.close, 6)) / sma(self.close, 6), 3), 12)

    def alpha023(self):
        cond_1 = self.close > delay(self.close, 1)
        cond_2 = self.close <= delay(self.close, 1)
        alpha = stddev(self.close, 20)
        alpha1 = alpha.copy()
        alpha2 = alpha.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return sma(alpha2, 20) / (sma(alpha1, 20) + sma(alpha2, 20))

    def alpha024(self):
        return sma(self.close - delay(self.close, 5), 5)

    def alpha025(self):
        return -1 * rank(delta(self.close, 7)) * (1 - rank(decay_linear(self.volume / sma(self.volume, 20), 9)) * (1 + rank(ts_sum(self.returns, 250))))

    def alpha026(self):
        return ts_sum(self.close, 7) / 7 - self.close + correlation(self.vwap, delay(self.close, 5), 230)

    def alpha027(self):
        return wma((self.close - delay(self.close, 3)) / delay(self.close, 3) * 100 + (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100, 12)

    def alpha028(self):
        return 3 * sma((self.close - delay(self.close, 3)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3) - 2 * sma(sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_max(self.low, 9)) * 100, 3), 3)

    def alpha029(self):
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume

    def alpha031(self):
        return (self.close - sma(self.close, 12) / sma(self.close, 12) * 100)

    def alpha032(self):
        return -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)

    def alpha033(self):
        return -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5) * rank(ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) * ts_rank(self.volume, 5)

    def alpha034(self):
        return sma(self.close, 12) / self.close

    def alpha035(self):
        return cross_min(rank(decay_linear(delta(self.open, 1), 15)), rank(decay_linear(correlation(self.volume, self.open * 0.65 + self.close * 0.35, 17), 7)) * -1)

    def alpha036(self):
        return rank(ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2))

    def alpha037(self):
        return -1 * ts_rank(ts_sum(self.open, 5) * ts_sum(self.returns, 5) - delay(ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)

    def alpha038(self):
        cond = ts_sum(self.high, 20) / 20 - self.high >= 0
        alpha = -1 * delta(self.high, 2)
        alpha[cond] = 0
        return alpha

    def alpha039(self):
        return -rank(decay_linear(delta(self.close, 2), 8)) + rank(decay_linear(correlation(0.3 * self.vwap + 0.7 * self.open, ts_sum(sma(self.volume, 180), 37), 14), 12))

    def alpha040(self):
        alpha1 = self.volume.copy()
        alpha2 = self.volume.copy()
        cond_1 = self.close <= delay(self.close, 1)
        cond_2 = self.close > delay(self.close, 1)
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
        return (sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close, 24)) / (self.close * 4)

    def alpha047(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9)

    def alpha048(self):
        return -1 * rank(sign(self.close - delay(self.close, 1)) + sign(delay(self.close, 1) - delay(self.close, 2)) + sign(delay(self.close, 2) - delay(self.close, 3))) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    def alpha049(self):
        cond_1 = self.high + \
            self.low >= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)),
                          abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0

        return ts_sum(alpha1, 12) / ts_sum(bound, 12)

    def alpha050(self):
        cond_1 = self.high + \
            self.low <= delay(self.high, 1) + delay(self.low, 1)
        cond_2 = self.high + \
            self.low >= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)),
                          abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha2 = bound.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return ts_sum(alpha1, 12) / ts_sum(bound, 12) - ts_sum(alpha2, 12) / ts_sum(bound, 12)

    def alpha051(self):
        cond_1 = self.high + \
            self.low <= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)),
                          abs(self.low - delay(self.low, 2)))
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

    def alpha060(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 20)

    def alpha062(self):
        return -1 * correlation(self.high, rank(self.volume), 5)

    def alpha063(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 6) / sma((self.close - delay(self.close, 1)).abs(), 6)

    def alpha064(self):
        return cross_max(rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4)), rank(decay_linear(ts_max(correlation(rank(self.close), rank(sma(self.volume, 60)), 4), 13), 14)))

    def alpha065(self):
        return sma(self.close) / self.close

    def alpha066(self):
        return (self.close - sma(self.close, 6)) / sma(self.close, 6) * 100

    def alpha067(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 24) / sma(np.abs(self.close - delay(self.close, 1)), 24) * 100

    def alpha068(self):
        return sma(((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2) * (self.high - self.low) / self.volume, 2)

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
        return ts_rank(self.volume / sma(self.volume, 20), 20) * ts_rank(-1 * delta(self.close, 7) / self.close, 8)

    def alpha087(self):
        return -1 * rank(decay_linear(delta(self.vwap, 4), 7)) + ts_rank(decay_linear(self.low - self.vwap) / (self.open - (self.low + self.high) / 2.0, 11), 7)

    def alpha088(self):
        return self.close / delay(self.close, 20) - 1

    def alpha090(self):
        return -1 * rank(correlation(rank(self.vwap), rank(self.volume), 5))

    def alpha091(self):
        return -1 * rank(self.close / ts_max(self.close, 5)) * rank(correlation(sma(self.volume, 40), self.low, 5))

    def alpha093(self):
        cond = self.open > delay(self.open, 1)
        alpha = cross_max(self.open - self.low,
                          self.open - delay(self.open, 1))
        alpha[cond] = 0
        alpha = alpha / self.close
        return ts_sum(alpha, 20)

    def alpha095(self):
        return stddev(self.amount, 20)

    def alpha096(self):
        return sma(sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)), 3), 3)

    def alpha097(self):
        return stddev(self.volume, 10)

    def alpha098(self):
        cond = delta(ts_sum(self.close, 100) / 100.0, 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha


    def alpha099(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha100(self):
        return stddev(self.volume, 20)

    def alpha102(self):
        return sma(cross_max(self.volume - delay(self.volume, 1), 0), 6) / sma((self.volume - delay(self.volume, 1)).abs(), 6)

    def alpha103(self):
        return (20 - lowday(self.low, 20) / 20)

    def alpha104(self):
        return -1 * delta(correlation(self.high, self.volume, 5), 5) * rank(stddev(self.close, 20))

    def alpha105(self):
        return -1 * correlation(rank(self.open), rank(self.volume), 10)

    def alpha106(self):
        return self.close - delay(self.close, 20)

    def alpha107(self):
        return -1 * rank(self.open - delay(self.high, 1)) * rank(self.open - delay(self.close, 1)) * rank(self.open - delay(self.low, 1))

    def alpha108(self):
        return rank(self.high - ts_min(self.high, 2)) ** rank(correlation(self.vwap), sma(self.volume, 120), 6) * -1

    def alpha109(self):
        return sma(self.high - self.low, 10) / sma(sma(self.high - self.low, 10), 10)

    def alpha110(self):
        return ts_sum(cross_max(0, self.high - delay(self.close, 1)), 20) / ts_sum(cross_max(0, delay(self.close, 1) - self.low), 20)

    def alpha111(self):
        return sma(self.volume * (self.close - self.low - self.high + self.close) / (self.high - self.low), 11) - sma(self.volume * (self.close - self.low - self.high + self.close) / (self.high - self.low), 4)

    def alpha112(self):
        cond_1 = self.close >= delay(self.close)
        cond_2 = self.close <= delay(self.close)
        alpha = self.close - delay(self.close)
        ts_1 = alpha.copy()
        ts_1[cond_2] = 0
        ts_2 = alpha.abs().copy()
        ts_2[cond_1] = 0
        return (ts_sum(ts_1, 12) - ts_sum(ts_2, 12)) / (ts_sum(ts_1, 12) + ts_sum(ts_2, 12))


    def alpha113(self):
        return -1 * rank(ts_sum(delay(self.close, 5), 20) / 20) * correlation(self.close, self.volume, 2) * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))

    def alpha114(self):
        return rank(delay((self.high - self.low) / ts_sum(self.close, 5) / 5), 2) * rank(rank(self.volume) / (self.high - self.low) / (ts_sum(self.close, 5) / 5) / (self.vwap - self.close))

    def alpha115(self):
        return rank(correlation(self.high * 0.9 + self.close * 0.1, sma(self.volume, 30), 10)) ** rank(correlation(ts_rank(self.high * 0.5 + self.low * 0.5, 4), ts_rank(self.volume, 10), 7))

    def alpha117(self):
        return ts_rank(self.volume, 32) * (1 - ts_rank((self.close + self.high - self.low), 16)) * (1 - ts_rank(self.returns, 32))

    def alpha118(self):
        return ts_sum(self.high - self.open, 20) / ts_sum(self.open - self.low, 20)

    def alpha119(self):
        return rank(decay_linear(correlation(self.vwap, ts_sum(sma(self.volume, 5), 26), 5), 7)) -  rank(decay_linear(ts_rank(ts_min(correlation(rank(self.open), rank(sma(self.volume, 15)), 21), 9), 7), 8))

    def alpha120(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha121(self):
        return rank(self.vwap - ts_min(self.vwap, 12)) ** ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(sma(self.volume, 60), 2), 18), 3) * -1

    def alpha122(self):
        return (sma(sma(sma(np.log(self.close), 13), 13), 13) - delay(sma(sma(sma(np.log(self.close), 13), 13), 13), 1)) / delay(sma(sma(sma(np.log(self.close), 13), 13), 13), 1)

    def alpha123(self):
        return rank(correlation(ts_sum(self.high * 0.5 + self.low * 0.5, 20), ts_sum(sma(self.volume, 60), 20),  9)) - rank(correlation(self.low, self.volume, 6))

    def alpha124(self):
        return (self.close - self.vwap) / decay_linear(rank(ts_max(self.close, 30)), 2)

    def alpha125(self):
        return rank(decay_linear(correlation(self.vwap, sma(self.volume, 80), 17), 20)) / rank(decay_linear(delta(self.close * 0.5 + self.vwap * 0.5, 3), 16))

    def alpha126(self):
        return (self.close + self.high + self.low) / 3

    def alpha127(self):
        return sma((self.close - ts_max(self.close, 12)) / ts_max(self.close, 12), 20)

    def alpha128(self):
        cond_1 = (self.high + self.low + self.close) >= delay(self.high + self.low + self.close)
        cond_2 = (self.high + self.low + self.close) <= delay(self.high + self.low + self.close)
        alpha = (self.high + self.low + self.close) / 3.0 * self.volume
        ts_1, ts_2 = alpha.copy(), alpha.copy()
        ts_1[cond_2] = 0
        ts_2[cond_1] = 0
        return 100 - (100 / (1 + ts_sum(ts_1, 14)/ ts_sum(ts_2, 14)))

    def alpha129(self):
        cond = self.close >= delay(self.close, 1)
        alpha = (self.close - delay(self.close, 1)).abs()
        alpha[cond] = 0
        return ts_sum(alpha, 12)

    def alpha130(self):
        return rank(decay_linear(correlation(self.high * 0.5 + self.low * 0.5, sma(self.volume, 40), 9), 10)) / rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 7), 3))

    def alpha131(self):
        return rank(delta(self.vwap, 1)) ** ts_rank(correlation(self.close, sma(self.volume, 50), 18), 18)

    def alpha132(self):
        return sma(self.amount, 20)

    def alpha133(self):
        return (20 - highday(self.high, 20) / 20) / 20 - (20 - lowday(self.low, 20) / 20)

    def alpha134(self):
        return (self.close - delay(self.close, 12)) / delay(self.close, 12) * self.volume

    def alpha135(self):
        return sma(delay(self.close / delay(self.close, 20, 1)), 20)

    def alpha136(self):
        return (-1 * rank(delta(self.returns, 3))) * correlation(self.open, self.volume, 10)

    def alpha138(self):
        return (rank(decay_linear(delta(self.low * 0.7 + self.vwap * 0.3, 3), 20) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(self.low, 8), ts_rank(sma(self.volume, 60), 17), 5), 19), 16), 7)))

    def alpha139(self):
        return -1 * correlation(self.open, self.volume, 10)

    def alpha140(self):
        return cross_min(rank(decay_linear(rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close), 8)), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(sma(self.volume, 60), 20), 8), 7), 3))

    def alpha141(self):
        return rank(correlation(rank(self.high), rank(sma(self.volume, 15)), 9)) * -1

    def alpha142(self):
        return -1 * rank(ts_rank(self.close, 10)) * rank(delta(delta(self.close))) * rank(ts_rank(self.volume / sma(self.volume, 20), 5))

    def alpha143(self):
        cond = self.close > delay(self.close)
        alpha = delay(self.alpha143())
        alpha[cond] = (self.close - delay(self.close)) / \
            delay(self.close) * self.alpha143()
        return alpha

    def alpha144(self):
        return sumif((self.close / delay(self.close) - 1).abs() / self.amount, 20, self.close < delay(self.close)) / count(self.close < delay(self.close), 20)

    def alpha145(self):
        return (sma(self.volume, 9) - sma(self.volume, 26)) / sma(self.volume, 12)

    def alpha148(self):
        return rank(correlation(self.open, ts_sum(sma(self.volume, 60), 9), 6)) - rank(self.open - ts_min(self.open, 14))

    def alpha150(self):
        return (self.close + self.high + self.low) / 3 * self.volume

    def alpha151(self):
        return sma(self.close - delay(self.close, 20), 20)

    def alpha152(self):
        return sma(sma(delay(sma(delay(self.close / delay(self.close, 9)),9)), 12) - sma(delay(sma(delay(self.close / delay(self.close, 9)), 9)), 26), 9)

    def alpha153(self):
        return (sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close, 24)) / 4

    def alpha158(self):
        return (self.high - self.low) / self.close

    def alpha160(self):
        cond = self.close > delay(self.close)
        alpha = stddev(self.close, 20)
        alpha[cond] = 0
        return sma(alpha, 20)

    def alpha161(self):
        return sma(cross_max(cross_max(self.high - self.low, np.abs(delay(self.close) - self.high)), np.abs(delay(self.close) - self.low)), 12)

    def alpha163(self):
        return rank(-1 * self.returns * sma(self.volume, 20) * self.vwap * (self.high - self.close))

    def alpha164(self):
        cond = self.close <= delay(self.close)
        alpha = 1 / (self.close - delay(self.close))
        alpha[cond] = 1
        return sma((alpha - ts_min(alpha, 12)) / (self.high - self.low), 13)

    def alpha167(self):
        cond = self.close <= delay(self.close)
        alpha = self.close - delay(self.close)
        alpha[cond] = 0
        return ts_sum(alpha, 12)

    def alpha168(self):
        return self.volume / sma(self.volume, 20) * -1

    def alpha169(self):
        return sma(sma(delay(sma(self.close - delay(self.close), 9)), 12) - sma(delay(sma(self.close - delay(self.close), 9)), 26), 10)

    def alpha170(self):
        return rank(1.0 / self.close) * self.volume / sma(self.volume, 20) * (self.high *rank(self.high - self.close)) / (ts_sum(self.high, 5) / 5) - rank(self.vwap - delay(self.vwap))

    def alpha171(self):
        return -1 * (self.low - self.close) * (self.open ** 5) / ((self.close - self.high) ** (self.close **５))

    def alpha174(self):
        return sumif(stddev(self.close, 20), 20, self.close > delay(self.close)) / 20.0

    def alpha175(self):
        return sma(cross_max(cross_max(self.high - self.low, np.abs(self.high - delay(self.close))), np.abs(delay(self.close) - self.low)), 6)

    def alpha176(self):
        return correlation(rank((self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12))), rank(self.volume), 6)

    def alpha177(self):
        return (20 - highday(self.high, 20) / 20) * 100

    def alpha178(self):
        return (self.close - delay(self.close)) / delay(self.close) * self.volume

    def alpha179(self):
        return rank(correlation(self.vwap, self.volume, 4)) * rank(correlation(rank(self.low), rank(sma(self.volume, 50)), 12))

    def alpha180(self):
        cond = sma(self.volume, 20) < self.volume
        alpha = -1 *self.volume
        alpha[cond] = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        return alpha

    def alpha184(self):
        return rank(correlation(delay(self.open - self.close), self.close, 200)) + rank(self.open - self.close)

    def alpha185(self):
        return rank(-1 * (1 - self.open / self.close) ** 2)

    def alpha187(self):
        alpha = cross_max(self.high - self.low, self.open - delay(self.open))
        alpha[self.open <= delay(self.open)] = 0
        return ts_sum(alpha, 20)

    def alpha188(self):
        return (self.high - self.low - sma(self.high - self.low, 11)) / sma(self.high - self.low, 11)

    def alpha189(self):
        return sma(np.abs(self.close - sma(self.close, 6)), 6)

    def alpha191(self):
        return correlation(sma(self.volume, 20), self.low, 5) + (self.high * 0.5 + self.low * 0.5 - self.close)
