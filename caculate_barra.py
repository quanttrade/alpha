import pandas as pd
import numpy as np
from barra_factor.py import *

if __name__ == "__main__":
    fundmental = pd.read_hdf('D:\data\daily_data\\ew.h5', 'ew')
    price_data = pd.read_hdf('D:\data\daily_data\\price_data.h5', 'table')
    pct_wdqa = pd.read_hdf('D:data/daily_data/pct_wdqa.h5', 'table')

    pn_data = pd.DataFrame()
    pn_data['LNCAP'] = np.log(fundmental[u'总市值'])
    pn_data['ETOP'] = fundmental['1/PE']
    pn_data['CETOP'] = (fundmental[u'每股现金净流量'].unstack() /
                        price_data['close']).stack()
    pn_data['SGRO5'] = fundmental['PSG5']
    pn_data['SGRO3'] = fundmental['PSG3']
    pn_data['EGRO5'] = fundmental['NPG5']
    pn_data['EGRO3'] = fundmental['NPG3']
    pn_data['BTOP'] = fundmental['1/PB']
    pn_data['MLEV'] = (fundmental[u'总市值'] +
                       fundmental[u'长期负债']) / fundmental[u'总市值']
    pn_data['DTOA'] = fundmental[u'总负债'] / fundmental[u'总资产']
    pn_data['BLEV'] = (fundmental[u'净资产'] +
                       fundmental[u'长期负债']) / fundmental[u'净资产']

    # NLSIZE
    standardize_cap = standardize(
        winsorize(pn_data['LNCAP'].copy().unstack(), mad_method))
    cap_cube = standardize_cap ** 3
    count = standardize_cap.count(axis=1)
    b = (count * (standardize_cap * cap_cube).sum(axis=1) - standardize_cap.sum(axis=1) *
         cap_cube.sum(axis=1)) / (count * (standardize_cap ** 2).sum(axis=1) - (standardize_cap.sum(axis=1))**2)
    pn_data['NLSIZE'] = (cap_cube - standardize_cap.multiply(b, axis=0)).stack()

    # Liquidity
    pn_data['STOM'] = np.log((price_data['volume'] / price_data['free_float_shares']
                             ).rolling(21).sum()).replace([-np.inf, np.inf], 0).stack()
    pn_data['STOQ'] = np.log(1 / 3.0 * (price_data['volume'] / price_data[
                             'free_float_shares']).rolling(63).sum()).replace([-np.inf, np.inf], 0).stack()
    pn_data['STOA'] = np.log(1 / 12.0 * (price_data['volume'] / price_data[
                             'free_float_shares']).rolling(242).sum()).replace([-np.inf, np.inf], 0).stack()

    # Beta
    pct_change = price_data['adjclose'].pct_change()
    beta, resid = beta_value(pct_change, pct_wdqa)
    pn_data['BETA'] = beta.stack()

    # Volatility
    Lambda_40 = np.power(0.5, 1 / 40.0)
    weight_40 = np.array([Lambda_40 ** (249 - i) for i in range(250)])
    excess_reeturns = (pct_change.T - pct_wdqa).T
    pn_data['DASTD'] = excess_reeturns.rolling(250).apply(
        lambda x: np.sqrt(np.average((x - x.mean(axis=0)) **
                          2, weights=weight_40, axis=0))).stack()

    def max_cumulative_returns(x):
        returns_array=np.array([x[-1] / x[-21 * T] for T in range(1, 13)])
        return returns_array.max(axis=0)

    def min_cumulative_returns(x):
        returns_array=np.array([x[-1] / x[-21 * T] for T in range(1, 13)])
        return returns_array.min(axis=0)

    pn_data['CMRA']=np.log(price_data['adjclose'].rolling(252).apply(
        lambda x: max_cumulative_returns(x) / min_cumulative_returns(x))).replace([-np.inf, np.inf], np.nan).stack()

    Lambda_60=np.power(0.5, 1 / 60.0)
    weight=pd.Series([Lambda_60 ** (pct_change.shape[0] - i - 1) for i in range(pct_change.shape[0])],
                       index=pct_change.index)
    weight_60=np.array([Lambda_60 ** (249 - i) for i in range(250)])
    hsigma=resid.divide(weight, axis=0).rolling(250).apply(
        lambda x: np.sqrt(np.average((x - x.mean(axis=0))
                          ** 2, weights=weight_60, axis=0)))
    pn_data['HSIGMA']=hsigma.stack()


    # Momentum
    Lambda_120=np.power(0.5, 1.0 / 120.0)
    weight_120=np.array([Lambda_120 ** (241 - i) for i in range(242)])
    momentum=pct_change.rolling(242 + 21).apply(lambda x: np.average(
        np.log(1 + x[:-21]), weights=weight_120, axis=0))
    pn_data['RSTR']=momentum.stack()



    # winsorize and standardize
    for descriptor in pn_data.columns:
        temp=pn_data[descriptor].copy().unstack()
        temp=standardize_cap(winsorize(temp, mad_method), cap)
        pn_data[descriptor]=temp.stack()

    #caculate_barra_factor
    barra_factor=pd.DataFrame()
    barra_factor['Beta']=pn_data['BETA']
    barra_factor['Momentum']=pn_data['RSTR']
    barra_factor['Size']=pn_data['LNCAP']
    barra_factor['Earning Yield']=0.21 * \
        pn_data['CETOP'] + 0.11 * pn_data['ETOP']
    barra_factor['Growth']=0.25 * pn_data['EGRO3'] + 0.25 * \
        pn_data['EGRO5'] + 0.25 * pn_data['SGRO5'] + 0.25 * pn_data['SGRO3']
    barra_factor['Leverge']=0.38 * pn_data['MLEV'] + \
        0.35 * pn_data['DTOA'] + 0.27 * pn_data['BLEV']
    barra_factor['NLSIZE']=pn_data['NLSIZE']
    barra_factor['Value']=pn_data['BTOP']
    barra_factor['Liquidity']=0.35 * pn_data['STOM'] + \
        0.35 * pn_data['STOQ'] + 0.3 * pn_data['STOA']
    barra_factor['Volatility']=0.74 * pn_data['DASTD'] + \
        0.16 * pn_data['CMRA'] + 0.1 * pn_data['HSIGMA']
