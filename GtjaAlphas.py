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
        self.adjclose = pn_data['adjclose']
        self.returns = self.adjclose.pct_change()
        self.amount = self.volume * self.close
        self.cap = pn_data['total_shares'] * pn_data['close']

    def gtja001(self):
        return -1 * correlation(rank(delta(log(self.volume), 1)), rank((self.close - self.open) / self.open), 6)

    def gtja002(self):
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low), 1)

    def gtja003(self):
        cond_1 = self.close == delay(self.close, 1)
        cond_2 = self.close > delay(self.close, 1)
        alpha = self.close - cross_max(self.high, delay(self.close, 1))
        alpha[cond_2] = self.close - cross_min(self.low, delay(self.close, 1))
        alpha[cond_1] = 0
        return ts_sum(alpha, 6)

    def gtja005(self):
        return -1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    def gtja007(self):
        return rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3)) * rank(delta(self.volume, 3))

    def gtja008(self):
        return rank(delta((self.high + self.low) / 2 * 0.2 + self.vwap * 0.8, 4) * -1)

    def gtja009(self):
        return sma(((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2) * ((self.high - self .low) / self.volume), 7)

    def gtja011(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 6)

    def gtja012(self):
        return rank(self.open - ts_sum(self.vwap, 10) / 10) * (-1 * (rank(abs(self.close - self.vwap))))

    def gtja013(self):
        return (self.high * self.low)**0.5 / self.vwap

    def gtja014(self):
        return self.close / delay(self.close, 5)

    def gtja015(self):
        return self.open / delay(self.close, 1) - 1

    def gtja016(self):
        return -1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def gtja017(self):
        return rank(self.vwap - ts_max(self.vwap, 15))**delta(self.close, 5)

    def gtja018(self):
        return self.close / delay(self.close, 5)

    def gtja019(self):
        cond_1 = self.close < delay(self.close, 5)
        cond_2 = self.close == delay(self.close, 5)
        alpha = (self.close - delay(self.close, 5)) / self.close
        alpha[cond_2] = 0
        alpha[cond_1] = (self.close - delay(self.close, 5)) / \
            delay(self.close, 5)
        return alpha

    def gtja020(self):
        return self.close / delay(self.close, 6) - 1

    def gtja022(self):
        return sma((self.close - sma(self.close, 6)) / sma(self.close, 6) - delay((self.close - sma(self.close, 6)) / sma(self.close, 6), 3), 12)

    def gtja023(self):
        cond_1 = self.close > delay(self.close, 1)
        cond_2 = self.close <= delay(self.close, 1)
        alpha = stddev(self.close, 20)
        alpha1 = alpha.copy()
        alpha2 = alpha.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return sma(alpha2, 20) / (sma(alpha1, 20) + sma(alpha2, 20))

    def gtja024(self):
        return sma(self.close - delay(self.close, 5), 5)

    def gtja025(self):
        return -1 * rank(delta(self.close, 7)) * (1 - rank(decay_linear(self.volume / sma(self.volume, 20), 9)) * (1 + rank(ts_sum(self.returns, 250))))

    def gtja027(self):
        return wma((self.close - delay(self.close, 3)) / delay(self.close, 3) * 100 + (self.close - delay(self.close, 6)) / delay(self.close, 6) * 100, 12)

    def gtja028(self):
        return 3 * sma((self.close - delay(self.close, 3)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3) - 2 * sma(sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_max(self.low, 9)) * 100, 3), 3)

    def gtja029(self):
        return (self.close - delay(self.close, 6)) / delay(self.close, 6) * self.volume

    def gtja031(self):
        return (self.close - sma(self.close, 12) / sma(self.close, 12) * 100)

    def gtja032(self):
        return -1 * ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)

    def gtja033(self):
        return -1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5) * rank(ts_sum(self.returns, 20) - ts_sum(self.returns, 20)) * ts_rank(self.volume, 5)

    def gtja034(self):
        return sma(self.close, 12) / self.close

    def gtja035(self):
        return cross_min(rank(decay_linear(delta(self.open, 1), 15)), rank(decay_linear(correlation(self.volume, self.open * 0.65 + self.close * 0.35, 17), 7)) * -1)

    def gtja036(self):
        return rank(ts_sum(correlation(rank(self.volume), rank(self.vwap), 6), 2))

    def gtja037(self):
        return -1 * ts_rank(ts_sum(self.open, 5) * ts_sum(self.returns, 5) - delay(ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)

    def gtja039(self):
        return -rank(decay_linear(delta(self.close, 2), 8)) + rank(decay_linear(correlation(0.3 * self.vwap + 0.7 * self.open, ts_sum(sma(self.volume, 180), 37), 14), 12))

    def gtja040(self):
        alpha1 = self.volume.copy()
        alpha2 = self.volume.copy()
        cond_1 = self.close <= delay(self.close, 1)
        cond_2 = self.close > delay(self.close, 1)
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        return ts_sum(alpha1, 26) / ts_sum(alpha2, 26)

    def gtja041(self):
        return rank(ts_max(delta(self.vwap, 3), 5)) * -1

    def gtja042(self):
        return -1 * rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def gtja043(self):
        cond_1 = self.close <= delay(self.close, 1)
        cond_2 = self.close >= delay(self.close)
        alpha1 = self.volume.copy()
        alpha2 = -self.volume.copy()
        alpha1[cond_1] = 0
        alpha2[cond_2] = 0
        alpha = alpha1 + alpha2
        return ts_sum(alpha, 2)

    def gtja044(self):
        return ts_rank(decay_linear(correlation(self.low, sma(self.volume, 10), 7), 6), 4) + ts_rank(decay_linear(delta(self.vwap, 3), 10), 15)

    def gtja045(self):
        return rank(delta(self.close * 0.6 + self.open * 0.4, 1)) * rank(correlation(self.vwap, sma(self.volume, 150), 15))

    def gtja046(self):
        return (sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close, 24)) / (self.close * 4)

    def gtja047(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)) * 100, 9)

    def gtja048(self):
        return -1 * rank(sign(self.close - delay(self.close, 1)) + sign(delay(self.close, 1) - delay(self.close, 2)) + sign(delay(self.close, 2) - delay(self.close, 3))) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    def gtja049(self):
        cond_1 = self.high + \
            self.low >= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)),
                          abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0

        return ts_sum(alpha1, 12) / ts_sum(bound, 12)

    def gtja050(self):
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

    def gtja051(self):
        cond_1 = self.high + \
            self.low <= delay(self.high, 1) + delay(self.low, 1)
        bound = cross_max(abs(self.high - delay(self.high, 1)),
                          abs(self.low - delay(self.low, 2)))
        alpha1 = bound.copy()
        alpha1[cond_1] = 0
        return ts_sum(alpha1, 12) / ts_sum(bound, 12)


    def gtja053(self):
        return (self.close > delay(self.close, 1)).rolling(12).sum()

    def gtja057(self):
        return sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)) * 100, 3)

    def gtja058(self):
        return (self.close > delay(self.close, 1)).rolling(20).sum() / 20 * 100

    def gtja059(self):
        alpha1 = cross_max(self.high, delay(self.close, 1))
        cond_1 = self.close > delay(self.close, 1)
        alpha1[cond_1] = cross_min(self.low, delay(self.close, 1))
        cond_2 = self.close == delay(self.close, 1)
        alpha2 = alpha1.copy()
        alpha2[cond_2] = 0
        return ts_sum(alpha2, 20)

    def gtja060(self):
        return ts_sum(((self.close - self.low) - (self.high - self.close)) / (self.high - self.low) * self.volume, 20)

    def gtja062(self):
        return -1 * correlation(self.high, rank(self.volume), 5)

    def gtja063(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 6) / sma((self.close - delay(self.close, 1)).abs(), 6)

    def gtja064(self):
        return cross_max(rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 4), 4)), rank(decay_linear(ts_max(correlation(rank(self.close), rank(sma(self.volume, 60)), 4), 13), 14)))

    def gtja065(self):
        return sma(self.close) / self.close

    def gtja066(self):
        return (self.close - sma(self.close, 6)) / sma(self.close, 6) * 100

    def gtja067(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 24) / sma(np.abs(self.close - delay(self.close, 1)), 24) * 100

    def gtja068(self):
        return sma(((self.high + self.low) / 2 - (delay(self.high, 1) + delay(self.low, 1)) / 2) * (self.high - self.low) / self.volume, 2)

    def gtja070(self):
        return stddev(self.amount.pct_change(), 6)

    def gtja071(self):
        return (self.close - sma(self.close, 24)) / (sma(self.close, 24)) * 100

    def gtja072(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_max(self.low, 6)), 15)

    def gtja073(self):
        return ts_rank(decay_linear(decay_linear(correlation(self.close, self.volume, 10), 16), 4), 5) - rank(decay_linear(correlation(self.vwap, sma(self.volume, 30), 4), 3))

    def gtja074(self):
        return rank(correlation(ts_sum(self.low * 0.35 + self.vwap * 0.65, 20), ts_sum(sma(self.volume, 40), 20), 7)) + rank(correlation(rank(self.vwap), rank(self.volume), 6))

    def gtja076(self):
        return stddev((self.close / delay(self.close) - 1).abs() / self.volume, 20) / sma((self.close / delay(self.close) - 1).abs() / self.volume, 20)


    def gtja078(self):
        return ((self.high + self.low + self.close) / 3.0 - sma((self.high + self.low + self.close) / 3.0, 12)) / sma((self.close - sma((self.high + self.low + self.close) / 3.0, 12)).abs(), 12)

    def gtja079(self):
        return sma(cross_max(self.close - delay(self.close, 1), 0), 12) / sma((self.close - delay(self.close, 1)).abs(), 12)

    def gtja080(self):
        return (self.volume - delay(self.volume, 5)) / delay(self.volume, 5)

    def gtja081(self):
        return sma(self.volume, 21) / self.volume

    def gtja082(self):
        return sma((ts_max(self.high, 6) - self.close) / (ts_max(self.high, 6) - ts_min(self.low, 6)), 20)

    def gtja083(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def gtja085(self):
        return ts_rank(self.volume / sma(self.volume, 20), 20) * ts_rank(-1 * delta(self.close, 7) / self.close, 8)

    def gtja088(self):
        return self.close / delay(self.close, 20) - 1

    def gtja090(self):
        return -1 * rank(correlation(rank(self.vwap), rank(self.volume), 5))

    def gtja091(self):
        return -1 * rank(self.close / ts_max(self.close, 5)) * rank(correlation(sma(self.volume, 20), self.low, 5))

    def gtja093(self):
        cond = self.open > delay(self.open, 1)
        alpha = cross_max(self.open - self.low,
                          self.open - delay(self.open, 1))
        alpha[cond] = 0
        alpha = alpha / self.close
        return ts_sum(alpha, 20)

    def gtja095(self):
        return stddev(self.amount, 20)

    def gtja096(self):
        return sma(sma((self.close - ts_min(self.low, 9)) / (ts_max(self.high, 9) - ts_min(self.low, 9)), 3), 3)

    def gtja097(self):
        return stddev(self.volume, 10)

    def gtja098(self):
        cond = delta(ts_sum(self.close, 100) / 100.0, 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha


    def gtja099(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def gtja100(self):
        return stddev(self.volume, 20)

    def gtja102(self):
        return sma(cross_max(self.volume - delay(self.volume, 1), 0), 6) / sma((self.volume - delay(self.volume, 1)).abs(), 6)


    def gtja104(self):
        return -1 * delta(correlation(self.high, self.volume, 5), 5) * rank(stddev(self.close, 20))

    def gtja105(self):
        return -1 * correlation(rank(self.open), rank(self.volume), 10)

    def gtja106(self):
        return self.close - delay(self.close, 20)

    def gtja107(self):
        return -1 * rank(self.open - delay(self.high, 1)) * rank(self.open - delay(self.close, 1)) * rank(self.open - delay(self.low, 1))


    def gtja109(self):
        return sma(self.high - self.low, 10) / sma(sma(self.high - self.low, 10), 10)

    def gtja110(self):
        return ts_sum(cross_max(0, self.high - delay(self.close, 1)), 20) / ts_sum(cross_max(0, delay(self.close, 1) - self.low), 20)

    def gtja111(self):
        return sma(self.volume * (self.close - self.low - self.high + self.close) / (self.high - self.low), 11) - sma(self.volume * (self.close - self.low - self.high + self.close) / (self.high - self.low), 4)

    def gtja112(self):
        cond_1 = self.close >= delay(self.close)
        cond_2 = self.close <= delay(self.close)
        alpha = self.close - delay(self.close)
        ts_1 = alpha.copy()
        ts_1[cond_2] = 0
        ts_2 = alpha.abs().copy()
        ts_2[cond_1] = 0
        return (ts_sum(ts_1, 12) - ts_sum(ts_2, 12)) / (ts_sum(ts_1, 12) + ts_sum(ts_2, 12))


    def gtja113(self):
        return -1 * rank(ts_sum(delay(self.close, 5), 20) / 20) * correlation(self.close, self.volume, 2) * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))

    def gtja115(self):
        return rank(correlation(self.high * 0.9 + self.close * 0.1, sma(self.volume, 30), 10)) ** rank(correlation(ts_rank(self.high * 0.5 + self.low * 0.5, 4), ts_rank(self.volume, 10), 7))

    def gtja117(self):
        return ts_rank(self.volume, 32) * (1 - ts_rank((self.close + self.high - self.low), 16)) * (1 - ts_rank(self.returns, 32))

    def gtja118(self):
        return ts_sum(self.high - self.open, 20) / ts_sum(self.open - self.low, 20)

    def gtja119(self):
        return rank(decay_linear(correlation(self.vwap, ts_sum(sma(self.volume, 5), 26), 5), 7)) -  rank(decay_linear(ts_rank(ts_min(correlation(rank(self.open), rank(sma(self.volume, 15)), 21), 9), 7), 8))

    def gtja120(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def gtja121(self):
        return rank(self.vwap - ts_min(self.vwap, 12)) ** ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(sma(self.volume, 60), 2), 18), 3) * -1

    def gtja122(self):
        return (sma(sma(sma(np.log(self.close), 13), 13), 13) - delay(sma(sma(sma(np.log(self.close), 13), 13), 13), 1)) / delay(sma(sma(sma(np.log(self.close), 13), 13), 13), 1)

    def gtja123(self):
        return rank(correlation(ts_sum(self.high * 0.5 + self.low * 0.5, 20), ts_sum(sma(self.volume, 60), 20),  9)) - rank(correlation(self.low, self.volume, 6))

    def gtja124(self):
        return (self.close - self.vwap) / decay_linear(rank(ts_max(self.close, 30)), 2)

    def gtja125(self):
        return rank(decay_linear(correlation(self.vwap, sma(self.volume, 80), 17), 20)) / rank(decay_linear(delta(self.close * 0.5 + self.vwap * 0.5, 3), 16))

    def gtja126(self):
        return (self.close + self.high + self.low) / 3

    def gtja127(self):
        return sma((self.close - ts_max(self.close, 12)) / ts_max(self.close, 12), 20)

    def gtja128(self):
        cond_1 = (self.high + self.low + self.close) >= delay(self.high + self.low + self.close)
        cond_2 = (self.high + self.low + self.close) <= delay(self.high + self.low + self.close)
        alpha = (self.high + self.low + self.close) / 3.0 * self.volume
        ts_1, ts_2 = alpha.copy(), alpha.copy()
        ts_1[cond_2] = 0
        ts_2[cond_1] = 0
        return 100 - (100 / (1 + ts_sum(ts_1, 14)/ ts_sum(ts_2, 14)))

    def gtja129(self):
        cond = self.close >= delay(self.close, 1)
        alpha = (self.close - delay(self.close, 1)).abs()
        alpha[cond] = 0
        return ts_sum(alpha, 12)

    def gtja130(self):
        return rank(decay_linear(correlation(self.high * 0.5 + self.low * 0.5, sma(self.volume, 40), 9), 10)) / rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 7), 3))

    def gtja131(self):
        return rank(delta(self.vwap, 1)) ** ts_rank(correlation(self.close, sma(self.volume, 50), 18), 18)

    def gtja132(self):
        return sma(self.amount, 20)

    def gtja133(self):
        return (20 - highday(self.high, 20) / 20) / 20 - (20 - lowday(self.low, 20) / 20)

    def gtja134(self):
        return (self.close - delay(self.close, 12)) / delay(self.close, 12) * self.volume

    def gtja136(self):
        return (-1 * rank(delta(self.returns, 3))) * correlation(self.open, self.volume, 10)

    def gtja139(self):
        return -1 * correlation(self.open, self.volume, 10)

    def gtja140(self):
        return cross_min(rank(decay_linear(rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close), 8)), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(sma(self.volume, 60), 20), 8), 7), 3))

    def gtja141(self):
        return rank(correlation(rank(self.high), rank(sma(self.volume, 15)), 9)) * -1

    def gtja142(self):
        return -1 * rank(ts_rank(self.close, 10)) * rank(delta(delta(self.close))) * rank(ts_rank(self.volume / sma(self.volume, 20), 5))


    def gtja144(self):
        return sumif((self.close / delay(self.close) - 1).abs() / self.amount, 20, self.close < delay(self.close)) / count(self.close < delay(self.close), 20)

    def gtja145(self):
        return (sma(self.volume, 9) - sma(self.volume, 26)) / sma(self.volume, 12)

    def gtja148(self):
        return rank(correlation(self.open, ts_sum(sma(self.volume, 60), 9), 6)) - rank(self.open - ts_min(self.open, 14))

    def gtja150(self):
        return (self.close + self.high + self.low) / 3 * self.volume

    def gtja151(self):
        return sma(self.close - delay(self.close, 20), 20)

    def gtja152(self):
        return sma(sma(delay(sma(delay(self.close / delay(self.close, 9)),9)), 12) - sma(delay(sma(delay(self.close / delay(self.close, 9)), 9)), 26), 9)

    def gtja153(self):
        return (sma(self.close, 3) + sma(self.close, 6) + sma(self.close, 12) + sma(self.close, 24)) / 4

    def gtja158(self):
        return (self.high - self.low) / self.close

    def gtja160(self):
        cond = self.close > delay(self.close)
        alpha = stddev(self.close, 20)
        alpha[cond] = 0
        return sma(alpha, 20)

    def gtja161(self):
        return sma(cross_max(cross_max(self.high - self.low, np.abs(delay(self.close) - self.high)), np.abs(delay(self.close) - self.low)), 12)

    def gtja163(self):
        return rank(-1 * self.returns * sma(self.volume, 20) * self.vwap * (self.high - self.close))

    def gtja164(self):
        cond = self.close <= delay(self.close)
        alpha = 1 / (self.close - delay(self.close))
        alpha[cond] = 1
        return sma((alpha - ts_min(alpha, 12)) / (self.high - self.low), 13)

    def gtja167(self):
        cond = self.close <= delay(self.close)
        alpha = self.close - delay(self.close)
        alpha[cond] = 0
        return ts_sum(alpha, 12)

    def gtja168(self):
        return self.volume / sma(self.volume, 20) * -1

    def gtja169(self):
        return sma(sma(delay(sma(self.close - delay(self.close), 9)), 12) - sma(delay(sma(self.close - delay(self.close), 9)), 26), 10)

    def gtja170(self):
        return rank(1.0 / self.close) * self.volume / sma(self.volume, 20) * (self.high *rank(self.high - self.close)) / (ts_sum(self.high, 5) / 5) - rank(self.vwap - delay(self.vwap))

    def gtja174(self):
        return sumif(stddev(self.close, 20), 20, self.close > delay(self.close)) / 20.0

    def gtja175(self):
        return sma(cross_max(cross_max(self.high - self.low, np.abs(self.high - delay(self.close))), np.abs(delay(self.close) - self.low)), 6)

    def gtja176(self):
        return correlation(rank((self.close - ts_min(self.low, 12)) / (ts_max(self.high, 12) - ts_min(self.low, 12))), rank(self.volume), 6)

    def gtja177(self):
        return (20 - highday(self.high, 20) / 20) * 100

    def gtja178(self):
        return (self.close - delay(self.close)) / delay(self.close) * self.volume

    def gtja179(self):
        return rank(correlation(self.vwap, self.volume, 4)) * rank(correlation(rank(self.low), rank(sma(self.volume, 50)), 12))

    def gtja180(self):
        cond = sma(self.volume, 20) < self.volume
        alpha = -1 *self.volume
        alpha[cond] = -1 * ts_rank(np.abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        return alpha

    def gtja184(self):
        return rank(correlation(delay(self.open - self.close), self.close, 200)) + rank(self.open - self.close)

    def gtja185(self):
        return rank(-1 * (1 - self.open / self.close) ** 2)

    def gtja187(self):
        alpha = cross_max(self.high - self.low, self.open - delay(self.open))
        alpha[self.open <= delay(self.open)] = 0
        return ts_sum(alpha, 20)

    def gtja188(self):
        return (self.high - self.low - sma(self.high - self.low, 11)) / sma(self.high - self.low, 11)

    def gtja189(self):
        return sma(np.abs(self.close - sma(self.close, 6)), 6)

    def gtja191(self):
        return correlation(sma(self.volume, 20), self.low, 5) + (self.high * 0.5 + self.low * 0.5 - self.close)


    # alpha002:(-1 * correlation(rank(delta(log(volume), 2)), rank(((close -
    # open) / open)), 6))

    def worldquant002(self):
        df = -1 * correlation(rank(delta(log(self.volume), 2)),
                              rank((self.close - self.open) / self.open), 6)
        return df.replace([-np.inf, np.inf], 0)


    # alpha004: (-1 * Ts_Rank(rank(low), 9))

    def worldquant004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def worldquant005(self):
        return rank(self.open - ts_sum(self.vwap, 10) / 10.0) * (-1 * abs(rank(self.close - self.vwap)))




    # alpha008: (-1 * rank(((sum(open, 5) * sum(returns, 5)) -
    # delay((sum(open, 5) * sum(returns, 5)),10))))

    def worldquant008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))

    # alpha009:((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
    # ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close,
    # 1))))

    def worldquant009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    # alpha010: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((t

    def worldquant010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def worldquant011(self):
        return rank(ts_max(self.vwap - self.close, 3)) + rank(ts_min(self.vwap - self.close, 3)) * rank(delta(self.volume, 3))

    #  alpha012:(sign(delta(volume, 1)) * (-1 * delta(close, 1)))

    def worldquant012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))


    # alpha017: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close,
    # 1), 1))) *rank(ts_rank((volume / adv20), 5)))

    def worldquant017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / adv20), 5)))

    # alpha018: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open))
    # + correlation(close, open,10))))

    def worldquant018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * (rank((stddev(abs((self.close - self.open)),
                                  5) + (self.close - self.open)) + df))

    #  alpha019:((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *

    def worldquant019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) +
                           delta(self.close, 7))) * (1 + rank(1 + ts_sum(self.returns, 250))))

    # alpha020: (((-1 * rank((open - delay(high, 1)))) * rank((open -
    # delay(close, 1)))) * rank((open -delay(low, 1))))

    def worldquant020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))


    # alpha024: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) <
    # 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) ==
    # 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))

    def worldquant024(self):
        cond = delta(sma(self.close, 100), 100) / \
            delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    # alpha026:(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5),
    # 5), 3))
    def worldquant026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0)
        return -1 * ts_max(df, 3)


    # alpha028:scale(((correlation(adv20, low, 5) + ((high + low) / 2)) -
    # close))

    def worldquant028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # alpha029:(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 *
    # rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 *
    # returns), 6), 5))
    def worldquant029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))),
                       5) + ts_rank(delay((-1 * self.returns), 6), 5))

    # alpha0230:(((1.0 - rank(((sign((close - delay(close, 1))) +
    # sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) -
    # delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))

    def worldquant030(self):
        delta_close = delta(self.close, 1).copy()
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + \
            sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / \
            ts_sum(self.volume, 20)

    # alpha031:((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close,
    # 10)))), 10)))) + rank((-1 *delta(close, 3)))) +
    # sign(scale(correlation(adv20, low, 12))))
    def worldquant031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12)
        df = df.replace([-np.inf, np.inf], 0)
        return ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10)))) +
                 rank((-1 * delta(self.close, 3)))) + sign(scale(df)))


    # alpha033: rank((-1 * ((1 - (open / close))^1)))

    def worldquant033(self):
        return rank(-1 + (self.open / self.close))

    # alpha034: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) +

    def worldquant034(self):
        inner = (stddev(self.returns, 2) / stddev(self.returns, 5)).copy()
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    # alpha035:((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low),
    # 16))) * (1 -Ts_Rank(returns, 32)))

    def worldquant035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))

    # alpha037:(rank(correlation(delay((open - close), 1), close, 200)) +
    # rank((open - close)))

    def worldquant037(self):
        return rank(correlation(delay(self.open - self.close, 1),
                                self.close, 200)) + rank(self.open - self.close)

    # alpha038: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))

    def worldquant038(self):
        inner = (self.close / self.open).copy()
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.open, 10)) * rank(inner)


    def worldquant042(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)



    # alpha049:(((((delay(close, 20) - delay(close, 10)) / 10) -
    # ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close
    # - delay(close, 1))))

    def worldquant049(self):
        inner = ((((delay(self.close, 20) - delay(self.close, 10)) /
                   10) - ((delay(self.close, 10) - self.close) / 10))).copy()
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha


    def worldquant052(self):
        return (-1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5) * rank(ts_sum(self.returns, 240) / 20.0 - ts_sum(self.returns, 20) / 20.0)) * ts_rank(self.volume, 5)

    # alpha052: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
    # rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume,
    # 5))

    # alpha053:(-1 * delta((((close - low) - (high - close)) / (close - low)),
    # 9))

    def worldquant053(self):
        inner = ((self.close - self.low).replace(0, 0.0001)).copy()
        return -1 * \
            delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # alpha054:((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    def worldquant054(self):
        inner = ((self.low - self.high).replace(0, -0.0001)).copy()
        return -1 * (self.low - self.close) * (self.open ** 5) / \
            (inner * (self.close ** 5))

    # alpha055: (-1 * correlation(rank(((close - ts_min(low, 12)) /
    # (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))

    def worldquant055(self):
        divisor = (
            ts_max(
                self.high,
                12) -
            ts_min(
                self.low,
                12)).replace(
            0,
            0.0001)
        inner = ((self.close - ts_min(self.low, 12)) / (divisor)).copy()
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0)

    def worldquant056(self):
        return (-1 * rank(ts_sum(self.returns, 10)) / ts_sum(ts_sum(self.returns, 2), 3)) * rank(self.returns * self.cap)

    def worldquant057(self):
        return (-1 * (self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2))

    # alpha060: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close))

    def worldquant060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = (((self.close - self.low) - (self.high - self.close))
                 * self.volume / divisor).copy()
        return - ((2 * scale(rank(inner))) -
                  scale(rank(ts_argmax(self.close, 10))))
