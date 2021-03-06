# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from barra_factor import *
from datetime import datetime
from WindPy import w
import statsmodels.api as sm
import ts_api_demo as ts
import pymysql
from WindPy import w
w.start()
from GtjaAlphas import *



def standardize_cap_day(factor, cap):
    cap = cap.ix[factor.index].fillna(0)
    return (factor - np.average(factor, weights=cap)) / factor.std()

def standardize_day_factor(factor):
    return (factor - factor.mean()) / factor.std()

def get_fundmental_day(date):
    return ts.get_barra_factor(date, date)


def create_daily_barra_factor(fundmental, price_data, benchmark_return, resid_return, date):


    #load the basic data of price and volume
    total_shares = price_data['total_shares'].ix[date][fundmental.index]
    volume = price_data['volume'].ix[:date][fundmental.index]
    free_float_shares = price_data['free_float_shares'].ix[:date][fundmental.index]
    adjclose = price_data['adjclose'].ix[:date][fundmental.index]
    cap = (price_data['total_shares'] * price_data['close']).ix[date][fundmental.index]
    resid_ret = resid_return.ix[:date].copy()
    for stk in fundmental.index:
        if stk not in resid_ret.columns:
            resid_ret[stk] = np.nan



    fundmental[u'净资产'] = total_shares * fundmental[u'每股净资产']
    fundmental[u'长期负债'] = total_shares * fundmental[u'每股长期负债']
    fundmental[u'总负债'] = total_shares * fundmental[u'每股负债']


    pn_data = pd.DataFrame()
    pn_data['LNCAP'] = np.log(fundmental[u'总市值'])
    pn_data['ETOP'] = fundmental['1/PE']
    pn_data['CETOP'] = (fundmental[u'每股现金净流量']/
                        price_data['close'].ix[date][fundmental.index])
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
    standardize_cap = standardize_day_factor(mad_method(pn_data['LNCAP']))
    cap_cube = standardize_cap ** 3
    count = standardize_cap.count()
    b = (count * (standardize_cap * cap_cube).sum() - standardize_cap.sum() *
         cap_cube.sum()) / (count * (standardize_cap ** 2).sum() - (standardize_cap.sum())**2)
    pn_data['NLSIZE'] = cap_cube - standardize_cap.multiply(b)


    pn_data['STOM'] = np.log((volume / free_float_shares)[-21:].rolling(21).sum()).replace([-np.inf, np.inf], 0).iloc[-1]
    pn_data['STOQ'] = np.log(1 / 3.0 * (volume / free_float_shares)[-63:].rolling(63).sum()).replace([-np.inf, np.inf], 0).iloc[-1]
    pn_data['STOA'] = np.log(1 / 12.0 * (volume / free_float_shares)[-252:].rolling(252).sum()).replace([-np.inf, np.inf], 0).iloc[-1]

    #beta
    beta = pd.Series(0.0, index=resid_ret.columns)
    resid_returns = pd.Series(0.0, index=resid_ret.columns)
    close_limit = price_data['adjclose'].ix[:date][-253:]
    stock_returns = close_limit.pct_change()[1:]
    Lambda_63 = np.power(0.5, 1 / 63.0)


    for stock in beta.index:
        returns_stock = stock_returns[stock].copy()
        volume_t = price_data['volume'].ix[:date][stock].ix[returns_stock.index]
        #returns_stock[volume_t == 0] = np.nan
        returns_stock = returns_stock.dropna()
        N = returns_stock.shape[0]
        if N < 126:
            beta[stock] = np.nan
            resid_returns[stock] = np.nan
            continue

        weight = pd.Series([Lambda_63 ** (N -1 -i) for i in range(N)], index=returns_stock.index)
        returns_stock_weight = returns_stock.multiply(weight)
        wdqa_pct = benchmark_return.ix[returns_stock.index].copy()
        wdqa_pct_weight = wdqa_pct.multiply(weight)
        wdqa_pct_weight = sm.add_constant(wdqa_pct_weight)
        wdqa_pct_weight.columns = ['const', 'beta']
        model_res = sm.OLS(returns_stock_weight, wdqa_pct_weight).fit()
        beta[stock] = model_res.params.beta
        resid_returns[stock] = model_res.params.const

    pn_data['BETA'] = beta[fundmental.index]

    for index in resid_returns.index:
        if index not in resid_ret.columns:
            resid_ret[index] = np.nan

    resid_ret.ix[date] = resid_returns
    resid_ret.index = pd.DatetimeIndex(resid_ret.index)
    resid_ret = resid_ret.sort_index()
    resid_ret.to_hdf('E:\multi_factor\\basic_factor\\resid_return.h5', 'table')


    # Volatility
    Lambda_42 = np.power(0.5, 1 / 42.0)
    weight_42 = np.array([Lambda_42 ** (252 - i) for i in range(252)])
    excess_return = (stock_returns[fundmental.index].T - benchmark_return.ix[stock_returns.index]).T
    excess_return_square = (excess_return - excess_return.mean(axis=0)) ** 2
    dastd = np.sqrt(np.average(excess_return_square, weights=weight_42, axis=0))
    pn_data['DASTD'] = dastd

    def max_cumulative_returns(x):
        returns_array=np.array([x[-1] / x[-21 * T] for T in range(1, 13)])
        return returns_array.max(axis=0)

    def min_cumulative_returns(x):
        returns_array=np.array([x[-1] / x[-21 * T] for T in range(1, 13)])
        return returns_array.min(axis=0)

    cmra = max_cumulative_returns(close_limit[fundmental.index].values) / min_cumulative_returns(close_limit[fundmental.index].values)\

    pn_data['CMRA'] = cmra

    Lambda_60=np.power(0.5, 1 / 63.0)
    weight_60=np.array([Lambda_60 ** (251 - i) for i in range(252)])
    resid = resid_ret.ix[-253:]
    hsigma = resid.rolling(252).apply(
        lambda x: np.sqrt(np.average((x - x.mean(axis=0))** 2, weights=weight_60, axis=0))).iloc[-1]
    pn_data['HSIGMA'] = hsigma.ix[fundmental.index]



    # Momentum
    Lambda_126=np.power(0.5, 1.0 / 126.0)
    weight_126=np.array([Lambda_126 ** (503 - i) for i in range(504)])
    weight_252 = np.array([Lambda_126 ** (251 - i) for i in range(252)])
    momentum_504 = pd.Series(np.average(np.log(adjclose[-505 - 21:].pct_change()[1:-21] + 1), weights=weight_126,axis=0), index=fundmental.index)
    momentum_252 = pd.Series(np.average(np.log(adjclose[-253 - 21:].pct_change()[1:-21] + 1), weights=weight_252,axis=0), index=fundmental.index)
    momentum = momentum_504.fillna(value=momentum_252)


    pn_data['RSTR'] = momentum
    industry_data = w.wsd(list(pn_data.index), "industry_citic", date, date, "industryType=1")
    if industry_data.ErrorCode != 0:
        industry_data = w.wss(list(pn_data.index), "industry_citic","tradeDate=%s;industryType=1" % date)
    industry_stock = pd.Series(industry_data.Data[0], index=industry_data.Codes)
    pn_data[u'行业'] = industry_stock

    pn_data_fill = pn_data.groupby(u'行业').apply(lambda x: x.fillna(x.quantile()))
    pn_data_fill.index = pn_data_fill.index.get_level_values(u'代码')
    del pn_data_fill[u'行业']
    pn_data_fill = pn_data_fill.apply(lambda x: standardize_cap_day(mad_method(x), cap))
    pn_data_fill[u'行业'] = pn_data[u'行业']
    pn_data = pn_data_fill.copy()


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

    vol = barra_factor['Volatility']
    size = barra_factor['Size']
    beta = barra_factor['Beta']

    beta_size = pd.concat([beta, size], axis=1)

    model = sm.OLS(vol, beta_size).fit()
    vol_resid = model.resid

    barra_factor['Volatility'] = vol_resid

    barra_factor = barra_factor.apply(lambda x: standardize_cap_day(x, cap))

    #pn_data[u'行业'] = map(lambda x: x[2:], pn_data[u'行业'])

    industry_set = list(set(pn_data[u'行业']))

    for industry in industry_set:
        temp = pd.Series(0.0, barra_factor.index)
        temp[pn_data[u'行业'] == industry] = 1.0
        barra_factor[industry] = temp

    return barra_factor


def get_basic_data(date, length, method='hdf'):
    begin_date = w.tdaysoffset(-length, date).Data[0][0]
    begin_date = ''.join(str(begin_date).split(' ')[0].split('-'))
    if method == 'database':
        conn = pymysql.connect(host='127.0.0.1',
                           port=3306,
                           user='root',
                           password='lyz940513',
                           db='mysql',
                           charset='utf8mb4',
                           cursorclass=pymysql.cursors.DictCursor)
        cursor = conn.cursor()

        cursor.execute('select distinct * from stockprice where tradedate<=%s and tradedate>=%s;' %(int(date), int(begin_date)))
        data = cursor.fetchall()
        data = pd.DataFrame(data)

    #close
        cursor.execute('select distinct * from stock_price where tradedate<=%s and tradedate>=%s;' %(int(date), int(begin_date)))
        prime_close = cursor.fetchall()
        prime_close = pd.DataFrame(prime_close)
        prime_close = prime_close.pivot(index='tradedate',columns='secid',values='prime_close')

        price_data = pd.Panel(load_data(data, prime_close))

    elif method == 'hdf':
        price_data = dict(pd.read_hdf('E:\multi_factor\price_data\price_data.h5', 'table'))

        for key in price_data.keys():
            data = price_data[key].copy()
            data = data.ix[:date]
            price_data[key] = data

        price_data = pd.Panel(price_data)






    # handle the fundmenl data from Tinysoft

    try:
        fundmental = get_fundmental_day(int(date))
    except Exception as e:
        print e


    if not fundmental.empty:

        fundmental_col = list(fundmental.columns)
        for i in range(len(fundmental_col)):
            fundmental_col[i] = fundmental_col[i].decode('utf-8')

        fundmental.columns = fundmental_col

        fundmental = fundmental[fundmental.st == 0]
        fundmental = fundmental[fundmental.pt == 0]

        fundmental[u'日期'] = map(str, fundmental[u'日期'])
        fundmental[u'日期'] = pd.DatetimeIndex(fundmental[u'日期'])
        fundmental[u'代码'] = map(lambda x: x[2:] + '.' + x[:2], fundmental[u'代码'])
        fundmental = fundmental.set_index(u'代码')


    else:
        date_before = w.tdaysoffset(-1, date).Data[0][0]
        date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
        fundmental = pd.read_hdf('E:\multi_factor\\basic_factor\\fundmental_%s.h5' % date_before, 'table')
        print "the data of %s is empty" % date




    volume = price_data['volume'].ix[:date]
    pct_wdqa = w.wsd('881001.WI', 'pct_chg', volume.index[0], volume.index[-1])
    pct_wdqa = pd.Series(pct_wdqa.Data[0], index=volume.index) / 100.0
    return fundmental, price_data, pct_wdqa


if __name__ == '__main__':
    dt = datetime.now()
    if dt.hour < 20:
        date_before = w.tdaysoffset(-1, dt).Data[0][0]
        date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
        dt = date_before
    date_list = map(lambda x: x.split('.')[0], os.listdir('E:/multi_factor/barra_factor'))
    if not date_list:
        last_date = '20070331'
    else:
        last_date = max(date_list)
    tradedate = w.tdays(last_date, dt).Data[0]
    tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)
    length = 540

    for date in tradedate[1:]:
        print date
        fundmental, price_data, pct_wdqa = get_basic_data(date, length)
        resid_ret = pd.read_hdf('E:\multi_factor\\basic_factor\\resid.return.h5','table')
        fundmental.to_hdf('E:\multi_factor\\basic_factor\\fundmental_%s.h5' % date, 'table')
        print fundmental[u'日期'][0]
        print price_data['close'].index[-1]
        barra_factor = create_daily_barra_factor(fundmental, price_data, pct_wdqa, resid_ret, date)
        barra_factor['COUNTRY'] = 1
        barra_factor.to_hdf('E:\multi_factor\\barra_factor\\%s.h5' % date, 'table')
        print barra_factor

"""
        Alpha = GtjaAlpha(price_data)
        alpha_function = GtjaAlpha.__dict__.keys()
        alpha_function.sort()
        alpha_function = alpha_function[5:]
        cap = price_data['close'] * price_data['total_shares']
        alpha_facor = pd.DataFrame()

        for alpha_name in alpha_function:
            print "========================caculating %s =============================" % alpha_name
            try:
                alpha = eval('Alpha.%s()' % alpha_name).dropna(how='all')

                # cleaning and standardize alpha data
                alpha = alpha.replace([-np.inf, np.inf], np.nan)

                alpha = standardize_cap_day(alpha.iloc[-1], cap.iloc[-1])

                alpha = alpha.ix[barra_factor.index]
                alpha = alpha.fillna(alpha.quantile())
                del barra_factor['COUNTRY']
                model = sm.OLS(alpha, barra_factor).fit()
                alpha_factor[alpha_name] = standardize_cap_day(model.resid, cap.iloc[-1])

        alpha_factor.to_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date, 'table')
        print alpha_factor
"""
