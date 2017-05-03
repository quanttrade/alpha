import talib as ta
import tushare as ts
import pandas as pd


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
        data['EMA5'] = ema_5
        data['EMA12'] = ema_12

        upperband, middleband, lowerband = ta.BBANDS(
            self.close, timeperiod=20)  # Bolling Bands
        data['UPPERBAND'] = upperband
        data['MIDDLEBAND'] = middleband
        data['LOWERBAND'] = lowerband

        CMO = ta.CMO(self.close)
        data['CMO'] = CMO

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

        self.data = data

    def technical_situation(self):
        data = self.data.dropna()
        situation = pd.DataFrame(index=data.index, columns=[
                                 'AROON', 'Kel', 'EMA', 'BOLL', 'CMO', 'SAR', 'MFI', 'RSI', 'MACD', 'TRIX', 'KD', 'DI', 'CCI', 'SAR'])
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
        situation = situation.fillna(0)
        after_index = situation.index[1:]
        situation = situation[:-1]
        situation.index = after_index
        situation['minus'] = situation.apply(
            lambda x: x[x == -1].shape[0], axis=1)
        situation['plus'] = situation.apply(
            lambda x: x[x == 1].shape[0], axis=1)
        self.situation = situation


def st(df):
    return df / df.iloc[0]


hs = ts.get_k_data('hs300', '2004-01-01', '2017-05-02')
hs = hs.set_index('date')
hs300 = hs.copy()
tt = technical_Analysis(hs300)
tt.technical_caculate()
tt.technical_situation()
mp = tt.situation.ix[:, -2:]
cha = mp.plus - mp.minus
pctchange = hs.close.pct_change()
returns = pd.DataFrame({'fuhao': cha, 'pct_chg': pctchange.ix[cha.index]})
returns_dk = returns.apply(lambda x: x.pct_chg if x[
                           0] > 1 else -x.pct_chg, axis=1)
returns_d = returns.apply(lambda x: x.pct_chg if x[0] > 1 else 0, axis=1)
net_dk = (1 + returns_dk).cumprod()
net_d = (1 + returns_d).cumprod()
net_dk = pd.DataFrame(
    {'strategy': net_dk, 'benchmark': st(hs.close.ix[net_dk.index])})
net_d = pd.DataFrame(
    {'strategy': net_d, 'benchmark': st(hs.close.ix[net_d.index])})
