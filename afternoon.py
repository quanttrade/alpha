import pandas as pd
import numpy as np
import os
from GtjaAlphas import *

if __name__ == '__main__':

    path = 'E:/minute_bar'
    dir_list = os.listdir(path)
    date_list = map(lambda x: x.split('.')[0], dir_list)
    hour_list = ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00']
    volume_ratio = dict()
    for filename in dir_list:
        data = pd.read_hdf(os.path.join(path, filename), 'table')
        grouped = data.groupby('time')
        date = filename.split('.')[0]
        date = date[:4] + '-' + date[4:6] +'-' + date[6:]
        time_list = map(lambda x: date + ' ' + x, hour_list)
        df_list = []
        for time in time_list:
            df = grouped.get_group(time)
            df = df.set_index('secid')
            del df['time']
            df_list.append(df)

        adjust_df = pd.DataFrame(index=df.index, columns=df.columns)


        volume_total = reduce(lambda x,y: x + y, [df.vol for df in df_list])
        volume_last = reduce(lambda x,y: x + y, [df.vol for df in df_list][-2:])
        ratio = (volume_last + 0.0) / volume_total
        volume_ratio[filename.split('.')[0]] = ratio



        adjust_df['open'] = df_list[0]['open']
        adjust_df['high'] = reduce(lambda x,y: cross_max(x, y), [df['high'] for df in df_list[:-2])
        adjust_df['low'] = reduce(lambda x,y: cross_min(x, y), df['low'] for df in df_list[:-2])
        adjust_df['close'] = df_list[5]['close']
        adjust_df['vwap'] = (reduce(lambda x,y: x + y, [df.amt for df in df_list[:-2]]) + 0.0) / reduce(lambda x,y: x+ y, [df.vol for df in df_list[:-2]])
        adjust_df['date'] = time_list[5]
