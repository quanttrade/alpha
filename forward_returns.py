# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from datetime import datetime
from WindPy import w
w.start()


if __name__ == '__main__':

    dt = datetime.now()
    if dt.hour < 16:
        date_before = w.tdaysoffset(-1, dt).Data[0][0]
        date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
        dt = date_before

    path = 'E:/multi_factor/barra_factor'
    periods = [2]
    price_data = pd.read_hdf('E:/multi_factor/price_data/price_data.h5', 'table')
    volume = price_data['volume']
    returns_dict = dict()
    for period in periods:
        returns_dict[period] = price_data['adjopen_vwap'].pct_change(period).shift(-period - 1)

    cap = price_data['close'] * price_data['total_shares']



    filename_list = os.listdir(path)
    date_list =  map(lambda x: x.split('.')[0], filename_list)


    #forward_returns = pd.DataFrame()
    choose = u'有色金属'
    for period in periods:
        returns = returns_dict[period]
        resid_returns = pd.read_hdf('E:\multi_factor\\forward_returns\\forward_returns_%s.h5' % period, 'table')
        if resid_returns.empty:
            begin_date = '2007-01-31'
        else:
            begin_date = resid_returns.index[-1]
        tradedate = w.tdays(begin_date, dt).Data[0]
        tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)
        for date in tradedate[: - period - 1]:
            print date

            # choose the stock tradable
            #import pdb; pdb.set_trace()
            volume_t = volume.ix[date]

            factor = pd.read_hdf(os.path.join(path, '%s.h5' % date), 'table')
            returns_t = returns.ix[date]
            factor = factor[volume_t > 0]
            factor = factor.replace([0], np.nan)
            factor = factor.dropna(how='all', axis=1)
            factor = factor.fillna(0)
            returns_t = returns_t.ix[factor.index]
            cap_t = cap.ix[date].ix[factor.index]
            industry_key = factor.columns[10:-1]

            # caculate the cap of every industry
            industry_set = factor[industry_key]
            industry_cap = pd.Series()
            for industry_name in industry_set.columns:
                industry_components = industry_set[industry_name]
                industry_components = industry_components[factor.index]
                industry_cap[industry_name] = cap_t[industry_components == 1].sum()

                # change the factor loading to satisfy w1 * f1 + w2 * f2 + ... wn * fn
                # = 0, wi, fi are industry cap and industry returns

            for name in industry_key:
                if name != choose:
                    factor[name] = factor[name] - industry_cap[name] / \
                    industry_cap[choose] * factor[choose]

            del factor[choose]

            # weighted regression to caculate the returns of each factor
            model = sm.WLS(returns_t.dropna(), factor.dropna(), weights=cap_t)

            try:
                res = model.fit()
                resid = res.resid
                for stk in resid.index:
                    if stk not in resid_returns.columns:
                        resid_returns[stk] = np.nan
                resid_returns.ix[date] = resid
            except Exception as e:
                print e
                continue
        resid_returns.index = pd.DatetimeIndex(resid_returns.index)
        resid_returns = resid_returns.fillna(0)
        print resid_returns
        resid_returns.to_hdf('E:\multi_factor\\forward_returns\\forward_returns_%s.h5' % str(period), 'table')
