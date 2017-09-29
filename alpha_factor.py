from GtjaAlphas import *
from barra_factor import *
import pandas as pd
import statsmodels.api as sm
from barra_factor_day import *
import os
from datetime import datetime
from WindPy import w
w.start()

dt = datetime.now()



if dt.hour < 16:
    date_before = w.tdaysoffset(-1, dt).Data[0][0]
    date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
    dt = date_before

def standardize_cap_day(factor, cap):
    cap = cap.ix[factor.index].fillna(0)
    return (factor - np.average(factor, weights=cap)) / factor.std()


if __name__ == "__main__":

    # get alpha function of GtjaAlpha

    alpha_function = GtjaAlpha.__dict__.keys()
    alpha_function.sort()
    alpha_function = alpha_function[5:]
    path = 'E:/multi_factor/alpha_factor'
    filename_list = os.listdir(path)
    date_list = map(lambda x: x.split('.')[0], filename_list)
    begin_date = date_list[-1]

    # load price and volume data
    price_data = dict(pd.read_hdf('E:\multi_factor\\price_data\\price_data.h5', 'table'))
    hour_list = ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00']

    for feature in price_data.keys():
        price_data[feature] = price_data[feature][-300:]

    price_data = pd.Panel(price_data)


    gtja = GtjaAlpha(price_data)
    cap = price_data['close'] * price_data['total_shares']
    tradedate = w.tdays(begin_date, dt).Data[0]
    tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)




    for alpha_name in alpha_function:
        print "========================caculating %s =============================" % alpha_name
        try:
            alpha = eval('gtja.%s()' % alpha_name).dropna(how='all')

            # cleaning and standardize alpha data
            alpha = alpha.replace([-np.inf, np.inf], np.nan)

            alpha = alpha.dropna(how='all', axis=1)
            alpha = alpha.dropna(how='all', axis=0)

            # import pdb; pdb.set_trace();

            for date in tradedate[1:]:
                print date
                cap_t = cap.ix[date]
                day_alpha = alpha.ix[date]
                barra_factor_day = pd.read_hdf('E:\multi_factor\\barra_factor\\%s.h5' % date,'table')
                del barra_factor_day['COUNTRY']
                day_alpha = day_alpha.ix[barra_factor_day.index]
                # import pdb; pdb.set_trace();
                day_alpha = day_alpha.fillna(day_alpha.quantile())
                day_alpha = standardize_cap_day(mad_method(day_alpha), cap_t)
                # import pdb; pdb.set_trace();
                model = sm.OLS(day_alpha, barra_factor_day).fit()
                # import pdb; pdb.set_trace();
                day_alpha_resid = model.resid
                # import pdb; pdb.set_trace();
                if os.path.exists('E:\multi_factor\\alpha_factor\\%s.h5' % date):
                    alpha_factor = pd.read_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date,'table')
                else:
                    alpha_factor = pd.DataFrame()
                alpha_factor[alpha_name] = day_alpha_resid

                alpha_factor[alpha_name] =standardize_cap_day(alpha_factor[alpha_name], cap.ix[date])
                alpha_factor.to_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date,'table')
                print alpha_factor


            del alpha

        except Exception as e:
            print e
            continue

    for date in tradedate[1:]:
        last_return_mean = pd.DataFrame()
        last_3_day = w.tdaysoffset(-3, date).Data[0][0]
        the_day_before = w.tdays(last_3_day, date).Data[0]
        the_day_before =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), the_day_before)
        for day in the_day_before:
            data = pd.read_hdf('E:\minute_bar\%s.h5' % date, 'table')
            grouped = data.groupby('time')
            date_line = date[:4] + '-' + date[4:6] +'-' + date[6:]
            time_list = map(lambda x: date_line + ' ' + x, hour_list)
            df_list = []
            for time in time_list:
                df = grouped.get_group(time)
                df = df.set_index('secid')
                del df['time']
                df_list.append(df)
            last_return_day = df_list[-1].close / df_list[-2].close - 1
            last_return_mean = pd.concat([last_return_mean, last_return_day],axis=1)
        day_alpha = last_return_mean.mean(axis=1)
        barra_factor_day = pd.read_hdf('E:\multi_factor\\barra_factor\\%s.h5' % date,'table')
        cap_t = cap.ix[date]
        del barra_factor_day['COUNTRY']
        day_alpha = day_alpha.ix[barra_factor_day.index]

        day_alpha = day_alpha.fillna(day_alpha.quantile())
        day_alpha = standardize_cap_day(mad_method(day_alpha), cap_t)
        model = sm.OLS(day_alpha, barra_factor_day).fit()
        day_alpha_resid = model.resid
        if os.path.exists('E:\multi_factor\\alpha_factor\\%s.h5' % date):
            alpha_factor = pd.read_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date,'table')
        else:
            alpha_factor = pd.DataFrame()
        alpha_factor['last_return'] = day_alpha_resid

        alpha_factor['last_return'] =standardize_cap_day(alpha_factor['last_return'], cap.ix[date])
        alpha_factor.to_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date,'table')
        print alpha_factor
