import numpy as np
import pandas as pd
import os
from scipy import stats
from datetime import datetime
from WindPy import w
w.start()

if __name__ == '__main__':

    dt = datetime.now()
    if dt.hour < 16:
        date_before = w.tdaysoffset(-1, dt).Data[0][0]
        date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
        dt = date_before


    periods = [2]
    for period in periods:
        path = 'E:\multi_factor\\alpha_factor'
        forward_returns = pd.read_hdf('E:\multi_factor\\forward_returns\\forward_returns_%s.h5' % period, 'table')
        ic = pd.read_hdf('E:\multi_factor\\adjusted_ic\\ic_%s.h5' % period, 'table')
        if ic.empty:
            begin_date = '20070131'
        else:
            begin_date = ic.index[-1]
        tradedate = w.tdays(begin_date, dt).Data[0]
        tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)


        for date in tradedate[1: - period - 1]:
            alpha_factor_date = pd.read_hdf(os.path.join(path, '%s.h5' % date), 'table')
            returns = forward_returns.ix[date].ix[alpha_factor_date.index].fillna(0)
            ic_date = alpha_factor_date.apply(lambda x: stats.pearsonr(x,returns)[0], axis=0)
            for alpha in ic_date.index:
                if alpha not in ic.columns:
                    ic[alpha] = np.nan
            ic.ix[date] = ic_date
            print date
        ic = ic.fillna(0)
        ic.index = pd.DatetimeIndex(ic.index)
        ic.to_hdf('E:\multi_factor\\adjusted_ic\\ic_%s.h5' % str(period), 'table')
