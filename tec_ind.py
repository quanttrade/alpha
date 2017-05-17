import talib as ta
import tushare as ts
import pandas as pd
import numpy as np
import itertools


def TD_index(high, low, m=5, k=1, p=3):
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
    for i in range(m):
        X.append(np.nan)

    # X_i = (hi - hi^k) + (li -li^k)  if hi >= li^m and li <=  hi^m else X_i = 0
    for i in range(m, length):
        if high[i] >= low[i - m : i].min()  and low[i] <= high[i - m : i].max():
            X_i = (high[i] - high[i - k:  i].max()) + (low[i] - low[i - k : i].min()) 

        else:
            X_i = 0

        X.append(X_i)

    df = pd.DataFrame({'high': high, 'low':low, 'momentum':X})
    
    # TD_index_i = (sum_j=0^p X(i-j)/ (hi^p - li^p)) *100
    momentum_sum = df.momentum.rolling(p).sum()
    standedlizer =  pd.Series([df.high[i - p : i].max() - df.low[i -p : i].min() for i in range(df.shape[0])])
    return (momentum_sum / standedlizer * 100).values



def emv(high, low, volume, N=14):
    a = (high + low) / 2
    b = []
    b.append(np.nan)
    b.extend(list(a[:-1]))
    b = np.array(b)
    c = high - low
    em = (a - b) * c / volume
    em = pd.Series(em)
    return em.rolling(N).mean().values



def psy(pctchange, N=12):
    pct_chg = pd.Series(pctchange)
    return pct_chg.rolling(N).apply(lambda x: x[x > 0].shape[0] / (x.shape[0] + 0.0) * 100).values


def tapi(pctchange, volume, N=24):
    data = pd.DataFrame({'pct_chg':pctchange, 'volume':volume})
    pass






class technical_Analysis:
    def __init__(self, data):
        self.data = data
        self.open = data.open.values
        self.high = data.high.values
        self.low = data.low.values
        self.close = data.close.values
        self.volume = data.volume.values
        self.pct_chg = self.data.close.pct_change()
        self.situation = pd.DataFrame()
        self.situation_pct = pd.DataFrame()

    def technical_caculate(self):
        data = self.data

        aroondown, aroonup = ta.AROON(self.high, self.low)
        aroon = aroonup - aroondown
        data['AROON'] = aroon

        atr = ta.ATR(self.high, self.low, self.close)

        ema_20 = ta.EMA(self.close, timeperiod=20)

        KeltnerLow = ema_20 - 2 * atr
        KeltnerHigh = ema_20 + 2 * atr
        data['KeltnerLow'] = KeltnerLow
        data['KeltnerHigh'] = KeltnerHigh

        ema_5 = ta.EMA(self.close, timeperiod=5)
        ema_12 = ta.EMA(self.close, timeperiod=12)
        ema_60 = ta.EMA(self.close, timeperiod = 60)
        data['EMA5'] = ema_5
        data['EMA12'] = ema_12
        data['EMA60'] = ema_60


        upperband, middleband, lowerband = ta.BBANDS(
            self.close, timeperiod=20)  # Bolling Bands
        data['UPPERBAND'] = upperband
        data['MIDDLEBAND'] = middleband
        data['LOWERBAND'] = lowerband

        CMO = ta.CMO(self.close)
        data['CMO'] = CMO

        MOM = ta.MOM(self.close)
        data['MOM'] = MOM

        CCI = ta.CCI(self.high, self.low, self.close)
        data['CCI'] = CCI

        SAR = ta.SAR(self.high, self.low)
        data['SAR'] = SAR

        MFI = ta.MFI(self.high, self.low, self.close, self.volume)
        data['MFI'] = MFI

        RSI = ta.RSI(self.close)
        data['RSI'] = RSI

        DIF, DEA, MACD = ta.MACD(self.close)
        data['DIF'] = DIF
        data['DEA'] = DEA
        data['MACD'] = MACD

        TRIX = ta.TRIX(self.close)
        data['TRIX'] = TRIX

        slowk, slowd = ta.STOCH(self.high, self.low, self.close)
        data['SLOWK'] = slowk
        data['SLOWD'] = slowd

        ADX = ta.ADX(self.high, self.low, self.close)
        PLUS_DI = ta.PLUS_DI(self.high, self.low, self.close)
        MINUS_DI = ta.MINUS_DI(self.high, self.low, self.close)
        data['ADX'] = ADX
        data['PLUS_DI'] = PLUS_DI
        data['MINUS_DI'] = MINUS_DI

        data['TD'] =  TD_index(self.high, self.low)

        data['VRSI'] = ta.RSI(self.volume, timeperiod = 6)

        data['EMV'] = emv(self.high, self.low, self.volume)

        data['PSY'] = psy(self.pct_chg)


        self.data = data

    def technical_situation(self):
        data = self.data.dropna()
        situation = pd.DataFrame(index=data.index, columns=[
                                 'AROON', 'Kel', 'EMA', 'BOLL', 'CMO', 'SAR', 'MFI', 'RSI', 'MACD', 'TRIX', 'KD', 'DI', 'CCI', 'MOM', 'TD', 'VRSI', 'EMV', 'PSY'])
        situation['AROON'].ix[data.AROON > 0] = 1
        situation['AROON'].ix[data.AROON < 0] = -1
        situation['Kel'].ix[data.close - data.KeltnerLow < 0] = -1
        situation['Kel'].ix[data.close - data.KeltnerHigh > 0] = 1
        situation['EMA'].ix[data.EMA5 - data.EMA12 > 0] = 1
        situation['EMA'].ix[data.EMA5 - data.EMA12 < 0] = -1
        situation['BOLL'].ix[data.close - data.LOWERBAND < 0] = -1
        situation['BOLL'].ix[data.close - data.UPPERBAND > 0] = 1
        situation['CMO'].ix[data.CMO < -50] = 1
        situation['CMO'].ix[data.CMO > 50] = -1
        situation['MFI'].ix[data.MFI > 80] = -1
        situation['MFI'].ix[data.MFI < 20] = 1
        situation['RSI'].ix[data.RSI > 70] = -1
        situation['RSI'].ix[data.RSI < 30] = 1
        situation['MACD'].ix[(data.DIF > 0) & (data.MACD > 0)] = 1
        situation['MACD'].ix[(data.DIF < 0) | (data.MACD < 0)] = -1
        situation['TRIX'].ix[data.TRIX > 0] = 1
        situation['TRIX'].ix[data.TRIX < 0] = -1
        situation['KD'].ix[(data.SLOWD > 90) | (data.SLOWK > 90)] = -1
        situation['KD'].ix[(data.SLOWD < 10) | (data.SLOWK < 10)] = 1
        situation['DI'].ix[data.PLUS_DI - data.MINUS_DI > 0] = 1
        situation['DI'].ix[data.PLUS_DI - data.MINUS_DI < 0] = -1
        situation['CCI'].ix[data.CCI < -100] = 1
        situation['CCI'].ix[data.CCI > 100] = -1
        situation['SAR'].ix[data.close - data.SAR > 0] = 1
        situation['SAR'].ix[data.close - data.SAR < 0] = -1
        situation['MOM'] .ix[data.MOM > 0] = 1
        situation['MOM'].ix[data.MOM < 0] = -1
        situation['TD'].ix[data.TD > 150] = 1
        situation['TD'].ix[data.TD < -150] = -1 
        situation['VRSI'].ix[data.VRSI > 85] = 1
        situation['VRSI'].ix[data.VRSI < 15] = -1
        situation['EMV'].ix[data.EMV > 0 ] = 1
        situation['EMV'].ix[data.EMV < 0 ] = -1
        situation['PSY'].ix[data.PSY > 80] = -1
        situation['PSY'].ix[data.PSY < 20] = 1

        situation = situation.fillna(0)
        self.situation = situation.copy()[1:]
        after_index = situation.index[1:]
        situation = situation[:-1]
        situation.index = after_index
        situation['minus'] = situation.apply(
            lambda x: x[x == -1].shape[0], axis=1)
        situation['plus'] = situation.apply(
            lambda x: x[x == 1].shape[0], axis=1)
        self.situation_pct = situation



def st(df):
    return df / df.iloc[0]


def return_sta(df):
    df = df.pct_change()
    grouped = df.groupby(lambda x: x.split('-')[0])
    return grouped.apply(lambda x: (1+x).prod() - 1)


def trade_indicator(ind, close):
    cash = 1.0
    p = 0
    netvalue = []
    buyprice = []
    sellprice = []
    buy_date = []
    sell_date = []
    for i in range(ind.shape[0]):
        date = ind.index[i]
        price = close.ix[date]
        if ind.ix[date] > 0 and cash > 0:
            p += cash / price
            cash = 0
            buyprice.append(price)
            buy_date.append(date)

        if ind.ix[date] < 0 and p > 0:
            cash += p * price
            p = 0
            sellprice.append(price)
            sell_date.append(date)

        netvalue.append(cash + price * p)

    df = pd.DataFrame({'netvalue' : netvalue, ' benchmark':  close.ix[ind.index[0]:].values}, index=ind.index)  

    if len(buyprice) > len(sellprice):
        buyprice = buyprice[ : -len(buyprice) + len(sellprice)]
        buy_date = buy_date[ :  -len(buy_date) + len(sell_date)]
                  #if there is one more buy action, delete it.
    trade_action = pd.DataFrame({'buyprice':buyprice, 'sellprice':sellprice, 'buydate':buy_date, 'selldate':sell_date})
    trade_action['returns'] = trade_action.sellprice / trade_action.buyprice -1
    return df / df.iloc[0] , trade_action



def enum_ret(situation, num, pctchange, start, end):
    N = len(situation.columns)
    value = 0
    choose_res = []
    result = list(itertools.permutations(range(N), num))
    for columns in result:
        df = situation.ix[: , columns].ix[start : end].copy()
        returns = caculate(df, pctchange)
        if returns > value:
            value = returns
            choose_res = columns
    return value,choose_res




def caculate(situation, pctchange):
    df = situation.sum(axis=1)
    return (pctchange.ix[df[df>0].index] + 1).prod()














    


hs = ts.get_k_data('hs300', '2004-01-01', '2017-05-02')
hs = hs.set_index('date')
hs300 = hs.copy()
tt = technical_Analysis(hs300)
tt.technical_caculate()
tt.technical_situation()
mp = tt.situation_pct.ix[:, -2:]
cha = mp.plus - mp.minus
pctchange = hs.close.pct_change()
returns = pd.DataFrame({'fuhao': cha, 'pct_chg': pctchange.ix[cha.index]})
returns_dk = returns.apply(lambda x: x.pct_chg if x[
                           0] > 1 else -x.pct_chg, axis=1)
returns_d = returns.apply(lambda x: x.pct_chg if x[0] > 1 else 0, axis=1)
net_dk = (1 + returns_dk).cumprod()
net_d = (1 + returns_d).cumprod()

net = pd.DataFrame({'net_d':net_d, 'net_dk':net_dk,'benchmark':st(hs.close.ix[net_dk.index])})
net_dk = pd.DataFrame(
    {'strategy': net_dk, 'benchmark': st(hs.close.ix[net_dk.index])})
net_d = pd.DataFrame(
    {'strategy': net_d, 'benchmark': st(hs.close.ix[net_d.index])})
