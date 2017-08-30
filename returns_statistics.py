import pandas as pd
import numpy as np

def maxdrawdown(net_value):
    draw_down = pd.Series([(net_value.iloc[i] - net_value[i:].min()) / net_value.iloc[i] for i in range(len(net_value) - 1)], index=net_value.index[:-1])

    maxdrawdown_value = draw_down.max()
    """
    maxdrawdown_index_start = draw_down.argmax()
    maxdrawdown_index_end = net_value[maxdrawdown_index_start:].argmin()
    """
    return maxdrawdown_value



def volatility(net_value):
    return net_value.pct_change().std() * np.sqrt(252.0)


def returns_(netvalue):
    return netvalue.iloc[-1] / netvalue.iloc[0] - 1

def statistics(netvalue):
    netvalue = netvalue.dropna()
    return netvalue.groupby(lambda x: x.year).agg([maxdrawdown, volatility, returns_])


def caculate_turnover(chicang):
    tradedate = chicang.index.get_level_values('date')
    tradedate = list(set(tradedate))
    tradedate.sort()
    turnover_Series = pd.Series(0.0, index=tradedate)
    for i in range(1, len(tradedate)):
        last_weight = chicang.ix[tradedate[i - 1]]
        now_weight = chicang.ix[tradedate[i]]
        stock_pool = list(set(last_weight.index) | set(now_weight.index))
        w_now = pd.Series(0.0, index=stock_pool)
        w_last = pd.Series(0.0, index=stock_pool)
        w_now.ix[now_weight.index] = now_weight
        w_last.ix[last_weight.index] = last_weight
        turnover_Series.ix[tradedate[i]] = (w_now - w_last).abs().sum()
    return turnover_Series[turnover_Series > 0] / 2.0
