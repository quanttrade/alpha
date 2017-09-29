import ts_api_demo as ts
from WindPy import w
import pandas as pd
import numpy as np
import os
w.start()

if __name__ == '__main__':
    tradedate = w.tdays('2005-01-04', '2017-09-11').Data[0]
    tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)


    path = 'E:/day_bar'
    """
    for date in tradedate:
        try:
            data = ts.get_adjprice(int(date))
            data['secid'] = data['secid'].apply(lambda x: x[2:]+'.'+x[:2])
            data.to_hdf(os.path.join(path , '%s.h5' % date), 'table')
            print date
        except Exception as e:
            print e
            continue
    """


    df = pd.DataFrame()



    for date in tradedate:
        try:
            data = pd.read_hdf(os.path.join(path, '%s.h5' % date), 'table')
            df = pd.concat([df, data])
            del data
        except Exception as e:
            print e
            continue
        print date

    df.to_hdf('E:/multi_factor/basic_factor/data.h5','table')
