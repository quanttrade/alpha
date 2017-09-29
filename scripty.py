import ts_api_demo as ts
from WindPy import w
import os
w.start()
from datetime import datetime


dt = datetime.now()



if dt.hour < 16:
    date_before = w.tdaysoffset(-1, dt).Data[0][0]
    date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
    dt = date_before

path = 'E:\minute_bar'
filename_list = os.listdir(path)
date_list = map(lambda x:x.split('.')[0], filename_list)
begin_date = date_list[-1]



tradedate = w.tdays(begin_date, dt).Data[0]
tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)
print tradedate



for date in tradedate[1:]:
    try:
        df = ts.get_minute_bar(int(date))
        df['secid'] = df['secid'].apply(lambda x: x[2:]+'.'+x[:2])
        df.to_hdf(os.path.join(path, '%s.h5' % date), 'table')
        print df
        print date
    except Exception as e:
        continue
