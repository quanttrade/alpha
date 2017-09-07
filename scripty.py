import ts_api_demo as ts
from WindPy import w
import os
w.start()

tradedate = w.tdays('2007-01-31', '2017-09-06').Data[0]
tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)
path = 'E:\minute_bar'


for date in tradedate:
    try:
        df = ts.get_minute_bar(int(date))
        df.to_hdf(os.path.join(path, '%s.h5' % date), 'table')
        print df
        print date
    except Exception as e:
        continue
