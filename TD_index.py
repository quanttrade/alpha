import pandas as pd
import  tushare as ts

def load_index_data(code, start, end):
    #load the index data
    df = ts.get_k_data(code, start, end)
    df = df.set_index('date')
    return df

def TD_index(high, low, m, k, p):
    """
    paramas:
    high : high price of  day
    low: low price of day

    returns:
    the td_index by guangfa
    """

    #k,m,p = 1,5,3
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
    standedlizer =  pd.Series([df.high[i - p : i].max() - df.low[i -p : i].min() for i in range(p , df.shape[0])], index=df.index[p:])
    return momentum_sum / standedlizer * 100


def trade_TD_index(TD, close, limit, fee):
    cash = 1.0
    p = 0
    netvalue = []
    buyprice = []
    sellprice = []
    buy_date = []
    sell_date = []
    for date in TD.index :
        if TD.ix[date] < -1 * limit and p > 0:
            cash += p * close.ix[date] * (1-fee)
            p = 0
            sellprice.append(close.ix[date])
            sell_date.append(date)

        if TD.ix[date] > limit and cash > 0:
            p += cash / close.ix[date] * (1-fee)
            cash = 0
            buyprice.append(close.ix[date])
            buy_date.append(date)

        netvalue.append(cash + p * close.ix[date])
    df = pd.DataFrame({'TD_varing' : netvalue, ' benchmark':  close.ix[TD.index[0]:].values}, index=TD.index)  

    if len(buyprice) > len(sellprice):
        buyprice = buyprice[: -len(buyprice) + len(sellprice)]
        buy_date = buy_date[:  -len(buy_date) + len(sell_date)]
    trade_action = pd.DataFrame({'buyprice':buyprice, 'sellprice':sellprice, 'buydate':buy_date, 'selldate':sell_date})
    trade_action['returns'] = trade_action.sellprice / trade_action.buyprice -1
    return df / df.iloc[0] , trade_action 


def TD_test(code, start, end, m, k , p, limit, fee):
    data = load_index_data(code, start, end)
    TD = TD_index(data.high, data.low, m, k, p)
    return trade_TD_index(TD, data.close, limit, fee)


    












