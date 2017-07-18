import pandas as pd
import numpy as np
from WindPy import w


def risk_parity(close):
    pct = close.pct_change().dropna()
    tradedate = w.tdays(pct.index[0], pct.index[-1], "Period=M")
    tradedate = map(lambda x: str(x).split(' ')[0], tradedate.Data[0])
    chicang = dict()
    for s in close.columns:
        chicang[s] = 0.0
    cash = 1.0
    stream = []
    weight = pd.DataFrame(index = pct.index[1:], columns=close.columns)
    for date in pct.index[1:]:
        iloc = list(pct.index).index(date)
        if date in tradedate:  
            if iloc >= 20:
                cov = pct[iloc - 20 : iloc].cov()
            else:
                cov = pct[: iloc].cov()
                

            for s in chicang.keys():
                cash += chicang[s] * close.ix[date][s]
                chicang[s] = 0.0

            sig = cov.values.diagonal() ** 0.5
            weight = sig ** -1 / (sig ** -1).sum()
            weight = pd.Series(weight, close.columns)

            cash1 = cash
            for s in chicang.keys():
                chicang[s] += weight[s] * cash1 / close.ix[date][s]
                cash -= cash1 * weight[s]
        weight.ix[date] = pd.Series(chicang)
        stream.append((pd.Series(chicang)*close.ix[date]).sum() + cash)
    return pd.Series(stream, index=pct.index[1:])





