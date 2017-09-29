# -*- coding: utf-8 -*-
from WindPy import w
import pymysql
from datetime import datetime
import numpy as np
import pandas as pd
import os
import ts_api_demo as ts
w.start()

dt = datetime.now()



if dt.hour < 16:
    date_before = w.tdaysoffset(-1, dt).Data[0][0]
    date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
    dt = date_before



path = 'E:/multi_factor'

#update benchmark_component
zz500_component = pd.read_hdf(os.path.join(path, 'benchmark_component', 'zz500_component.h5'), 'table')
l_date = zz500_component.index[-1]
hour_list = ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00']

tradedate = w.tdays(l_date, dt).Data[0]
tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)

for date in tradedate:
    print date
    data =  w.wset("sectorconstituent","date=%s;sectorid=1000008491000000" % date).Data[1]
    zz500_component.ix[date] = data

zz500_component.index = pd.DatetimeIndex(zz500_component.index)
zz500_component.to_hdf(os.path.join(path, 'benchmark_component', 'zz500_component.h5'), 'table')



# update price_data
price_data = dict(pd.read_hdf(os.path.join(path, 'price_data', 'price_data.h5'), 'table'))
last_date = price_data['close'].index[-1]

tradedate = w.tdays(last_date, dt).Data[0]
tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)
columns = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'amt', 'total_shares', 'free_float_shares', 'buy', 'sell', 'adjfactor']
for date in tradedate[1:]:
    try:
        print "update the data of %s" % date

        data = ts.get_adjprice(int(date))
        data['secid'] = data['secid'].apply(lambda x: x[2:]+'.'+x[:2])
        data = data.set_index('secid')


        for cols in columns:
            data_col = price_data[cols]
            insert_data = data[cols]
            for stks in insert_data.index:
                if stks not in data_col.columns:
                    if cols != 'adjfactor':
                        data_col[stks] = np.nan
                    else:
                        data_col[stks] = 1
            data_col.ix[date] = insert_data
            data_col.index = pd.DatetimeIndex(data_col.index)
            data_col = data_col.fillna(method='pad')
            price_data[cols] = data_col

        adjfactor = price_data['adjfactor']
        pre_adjfactor = adjfactor / adjfactor.iloc[-1]

        data = pd.read_hdf('E:\minute_bar\%s.h5' % date, 'table')
        open_vwap = price_data['open_vwap']
        grouped = data.groupby('time')
        date_line = date[:4] + '-' + date[4:6] +'-' + date[6:]
        time_list = map(lambda x: date_line + ' ' + x, hour_list)
        df = grouped.get_group(time_list[0])
        df = df.set_index('secid')
        open_vwap_td = df.vwap
        for s in open_vwap_td.index:
            if s not in open_vwap.columns:
                open_vwap[s] = np.nan
        open_vwap.ix[date] = open_vwap_td
        open_vwap.index = pd.DatetimeIndex(open_vwap.index)
        open_vwap = open_vwap.replace([0.0], np.nan)
        open_vwap = open_vwap.fillna(method='pad')
        price_data['open_vwap'] = open_vwap
        adj_columns = ['adjopen', 'adjhigh', 'adjlow', 'adjclose', 'adjvwap', 'adjvolume', 'adjopen_vwap']

        for cols in adj_columns:
            price_data[cols] = price_data[cols[3:]] * pre_adjfactor

        # import pdb; pdb.set_trace();
    except Exception as e:
        print e
# import pdb; pdb.set_trace();
pd.Panel(price_data).to_hdf(os.path.join(path, 'price_data', 'price_data.h5'), 'table')
