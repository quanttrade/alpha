from GtjaAlphas import *
from barra_factor import *
import pandas as pd
import statsmodels.api as sm
from barra_factor_day import *
if __name__ == "__main__":

    # get alpha function of GtjaAlpha

    alpha_function = GtjaAlpha.__dict__.keys()
    alpha_function.sort()
    alpha_function = alpha_function[5:]

    # load price and volume data
    price_data = dict(pd.read_hdf('E:\multi_factor\price_data\price_data.h5', 'table'))

    for key in price_data.keys():
        data = price_data[key].copy()
        data = data[-500:]
        price_data[key] = data

    price_data = pd.Panel(price_data)

    gtja = GtjaAlpha(price_data)
    cap = price_data['close'] * price_data['total_shares']
    begin_date = '2007-01-31'
    tradedate = w.tdays('2017-01-01', '2017-09-01').Data[0]
    tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)


    for alpha_name in alpha_function[:155]:
        print "========================caculating %s =============================" % alpha_name
        try:
            alpha = eval('gtja.%s()' % alpha_name).dropna(how='all')

            # cleaning and standardize alpha data
            alpha = alpha.replace([-np.inf, np.inf], np.nan)

            alpha = standardize_cap(alpha, cap)


            for date in tradedate:
                print date
                day_alpha = alpha.ix[date]
                barra_factor_day = pd.read_hdf('E:\multi_factor\\barra_factor\\%s.h5' % date,'table')
                del barra_factor_day['COUNTRY']
                day_alpha = day_alpha.ix[barra_factor_day.index]
                day_alpha = day_alpha.fillna(day_alpha.quantile())
                model = sm.OLS(day_alpha, barra_factor_day).fit()
                day_alpha_resid = model.resid
                alpha_facor = pd.read_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date,'table')
                alpha_facor[alpha_name] = day_alpha_resid
                alpha_facor[alpha_name] =standardize_cap_day(alpha_facor[alpha_name], cap.ix[date])
                alpha_facor.to_hdf('E:\multi_factor\\alpha_factor\\%s.h5' % date,'table')


            del alpha

        except Exception as e:
            print e