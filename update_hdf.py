from WindPy import w
import pymysql
from datetime import datetime
import numpy as np
import pandas as pd
import os


dt = datetime.now()
w.start()


if dt.hour < 16:
    date_before = w.tdaysoffset(-1, dt).Data[0][0]
    date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
    dt = date_before



path = 'E:/multi_factor'

#update benchmark_component
zz500_component = pd.read_hdf(os.path.join(path, 'benchmark_component', 'zz500_component.h5'), 'table')
l_date = zz500_component.index[-1]

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

for date in tradedate[1:]:
    try:
        print "update the data of %s" % date
        wsetdata = w.wset(
            'SectorConstituent',
            'date=%s;sectorId=a001010100000000;field=wind_code' % date).Data[0]

        wssdata = w.wss(wsetdata, "open,high,low,close,vwap,volume,amt,free_float_shares,total_shares","tradeDate=%s;priceAdj=F;cycle=D;unit=1" % date)
        wssdata_prime = w.wss(wsetdata, "close","tradeDate=%s;priceAdj=U;cycle=D" %date)

        adj_data = pd.DataFrame(np.array(wssdata.Data).T, index=wsetdata, columns="adjopen,adjhigh,adjlow,adjclose,adjvwap,volume,amt,free_float_shares,total_shares".split(','))
        adj_data['close'] = wssdata_prime.Data[0]
        adjust_factor = adj_data['close'] / adj_data['adjclose']
        adj_data['high'] = adj_data['adjhigh'] * adjust_factor
        adj_data['low'] = adj_data['adjlow'] * adjust_factor
        adj_data['open'] = adj_data['adjopen'] * adjust_factor
        adj_data['vwap'] = adj_data['adjvwap'] * adjust_factor

        for col in adj_data.columns:
            data_col = price_data[col]
            for stk in adj_data[col].index:
                if stk not in data_col.columns:
                    data_col[stk] = np.nan
            data_col.ix[date] = adj_data[col]
            price_data[col] = data_col

    except Exception as e:
        print e



for item in price_data.keys():
    data = price_data[item]
    data.index = pd.DatetimeIndex(data.index)
    price_data[item] = data



pd.Panel(price_data).to_hdf(os.path.join(path, 'price_data', 'price_data.h5'), 'table')
