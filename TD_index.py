import pandas as pd


def TD_index(high, low):
    """
    paramas:
    high : high price of  day
    low: low price of day

    returns:
    the td_index by guangfa
    """

    k,m,p = 2,6,5
    # tmodel's paramaters set to be as the begining
    length = len(high)
    X = []    # initialize the the daily momentum

    # X_i = (hi - hi^k) + (li -li^k)  if hi >= li^m and li <=  hi^m else X_i = 0
    for i in range(m, length):
        if high.iloc[i] >= low[i - m : i].min()  and low.iloc[i] <= high[i - m : i].max():
            X_i = (high.iloc[i] - high[i - k:  i].max()) + (low.iloc[i] - low[i - k : i].min()) 

        else:
            X_i = 0

        X.append(X_i)

    df = pd.DataFrame({'high': high[m:], 'low':low[m:], 'momentum':X}, index=high.index[m:])
    
    # TD_index_i = (sum_j=0^p X(i-j)/ (hi^p - li^p)) *100
    momentum_sum = df.momentum.rolling(p).sum().dropna()
    standedlizer =  pd.Series([high[i - p : i].max() - low[i -p : i].min() for i in range(p , df.shape[0])], index=df.index[p:])
    return momentum_sum / standedlizer * 100


def trade_TD_index(TD, close, limit):
    cash = 1.0
    p = 0
    fee = 0.0015
    netvalue = []
    for date in TD.index :
        if TD.ix[date] < -1 * limit:
            cash += p * close.ix[date] * (1-fee)
            p = 0

        if TD.ix[date] > limit:
            p += cash / close.ix[date] * (1-fee)
            cash = 0

        netvalue.append(cash + p * close.ix[date])
    df = pd.DataFrame({'TD_varing' : netvalue, ' benchmark':  close.ix[TD.index[0]:].values}, index=TD_index)
    return df / df.iloc[0]








