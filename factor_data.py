import alphalens
import pandas as pd
from scipy import stats
import numpy as np
from GtjaAlphas import *



def load_data(data):
    data['tradedate'] = data.tradedate.apply(str)
    data['tradedate'] = pd.DatetimeIndex(data.tradedate)

    data = data[data.tradedate > '2007-01-15']

    close = data.pivot(index='tradedate',
                       columns='secid',
                       values='closeprice')

    open_ = data.pivot(index='tradedate',
                       columns='secid',
                       values='openprice')

    high = data.pivot(index='tradedate',
                      columns='secid',
                      values='highprice')

    low = data.pivot(index='tradedate',
                     columns='secid',
                     values='lowprice')

    volume = data.pivot(index='tradedate',
                        columns='secid',
                        values='volume')

    total_shares = data.pivot(index='tradedate',
                              columns='secid',
                              values='total_shares')

    vwap = data.pivot(index='tradedate',
                      columns='secid',
                      values='vwap')

    amt = data.pivot(index='tradedate',
                     columns='secid',
                     values='amt')

    pn_data = dict()

    pn_data['close'] = close
    pn_data['high'] = high
    pn_data['low'] = low
    pn_data['open'] = open_
    pn_data['volume'] = volume
    pn_data['total_shares'] = total_shares
    pn_data['vwap'] = vwap
    pn_data['amt'] = amt

    return data, pn_data


def format_factor(factor):
    """
    params: factor formating like DataFrame
    returns: factor formating like Series
    """

    it = [list(pd.DatetimeIndex(factor.index)), list(factor.columns)]
    index = pd.MultiIndex.from_product(it, names=['tradeDate', 'secID'])
    factor_data = pd.DataFrame(factor.values.reshape(-1, 1), index=index)[0]
    return factor_data.dropna()


def handle_factor(factor, prices, groupby, periods, path):
    factor_format = format_factor(factor)
    prices_format = prices.ix[factor_format.index[0][0]:]



    # standard factor performance

    factor_data_standard = alphalens.utils.get_clean_factor_and_forward_returns(factor_format,
                                                                                prices_format,
                                                                                periods=periods)

    quantile_returns_mean_standard, quantile_returns_std_standard = alphalens.performance.mean_return_by_quantile(
        factor_data_standard)

    ic_standard = alphalens.performance.factor_information_coefficient(
        factor_data_standard)

    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_standard.mean()
    ic_summary_table["IC Std."] = ic_standard.std()
    t_stat, p_value = stats.ttest_1samp(ic_standard, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_standard)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_standard)
    ic_summary_table["Ann. IR"] = (
        ic_standard.mean() / ic_standard.std()) * np.sqrt(252)


    
    factor.to_excel(path + '//prime_factor.xlsx')
    factor_data_standard.to_excel(path + '//factor_data_standard.xlsx')
    quantile_returns_mean_standard.to_excel(path + '//quantile_returns_mean_standard.xlsx')
    ic_standard.to_excel(path + '//ic_standard.xlsx')
    ic_summary_table.to_excel(path + '//ic_summary_table.xlsx')




    # factor performance in different group(by industry, size, value ...etc)

    def information_statistcs(ic):              #caculate ic_statitic use the daily ic data
        ic_group = ic.groupby(ic.index.get_level_values('group'))
        ic_mean = ic_group.agg(np.mean)
        ic_std = ic_group.agg(np.std)
        ic_tsta = ic_group.agg(lambda x: (stats.ttest_1samp(x, 0)[0]))
        ic_pvalue = ic_group.agg(lambda x: (stats.ttest_1samp(x, 0)[1]))
        ic_skew = ic_group.agg(stats.skew)
        ic_kurtosis = ic_group.agg(stats.kurtosis)
        ic_ir = ic_mean / ic_std * np.sqrt(252)

        ic_mean.index = map(lambda x: x + '_ic_mean', ic_mean.index)
        ic_std.index = map(lambda x: x + '_ic_std', ic_std.index)
        ic_tsta.index = map(lambda x: x + '_ic_tsta', ic_tsta.index)
        ic_pvalue.index = map(lambda x: x + '_ic_pvalue', ic_pvalue.index)
        ic_skew.index = map(lambda x: x + '_ic_skew', ic_skew.index)
        ic_kurtosis.index = map(
            lambda x: x + '_ic_kurtosis', ic_kurtosis.index)
        ic_ir.index = map(lambda x: x + '_ic_ir', ic_ir.index)

        return pd.concat([ic_mean, ic_std, ic_tsta, ic_pvalue, ic_skew, ic_kurtosis, ic_ir])

    for key in groupby.keys():

        #generate the factor_data with forward returns and group

        factor_data_key = alphalens.utils.get_clean_factor_and_forward_returns(factor_format,
                                                                               prices_format,
                                                                               periods=periods,
                                                                               groupby=groupby[
                                                                                   key],
                                                                               by_group=True)

        #caculate factor returns with group_neutral

        factor_returns = alphalens.performance.factor_returns(
            factor_data_key, group_neutral=True)

        #caculate quantile returns_mean

        quantile_returns_mean_key, quantile_returns_std_key = alphalens.performance.mean_return_by_quantile(
            factor_data_key, by_group=True)

        # caculate ic by group

        ic = alphalens.performance.factor_information_coefficient(
            factor_data_key, by_group=True, group_adjust=True)

        ic_table = information_statistcs(ic)

        # turnover data

        turnover_periods = alphalens.utils.get_forward_returns_columns(
            factor_data_key.columns)
        quantile_factor = factor_data_key['factor_quantile']

        quantile_turnover = {p: pd.concat([alphalens.performance.quantile_turnover(
            quantile_factor, q, p) for q in range(1, int(quantile_factor.max()) + 1)],
            axis=1)
            for p in turnover_periods}


        quantile_turnover_mean = pd.Panel(quantile_turnover).mean()


        #save data to excel
        factor_data_key.to_excel(path + '//factor_data_%s.xlsx' % key)
        factor_returns.to_excel(path + '//factor_returns_%s.xlsx' % key)
        quantile_returns_mean_key.to_excel(path + '//quantile_returns_mean_%s.xlsx' % key)
        ic.to_excel(path + '//ic_%s.xlsx' % key)
        ic_table.to_excel(path + '//ic_table_%s.xlsx' % key)
        quantile_turnover_mean.to_excel(path + '//quantile_turnover_mean_%s.xlsx' % key)



if __name__ == "__main__":


    #get alpha function of GtjaAlpha

    alpha_function = GtjaAlpha.__dict__.keys()
    alpha_function.sort()
    alpha_function = alpha_function[5:]

    # load price and volume data

    data = pd.read_csv('/Users/liyizheng/data/daily_data/stock_data.csv')
    data,pn_data = load_data(data)
    gtja = GtjaAlpha(pn_data)


    # get class signal by hs300 and zz500
    hs300 = pd.read_csv('/Users/liyizheng/data/daily_data//hs300_component.csv',index_col=0)
    zz500 = pd.read_csv('/Users/liyizheng/data/daily_data//zz500_component.csv',index_col=0)
    stock = [list(pn_data['volume'].columns) for i in range(hs300.shape[0])]
    stock = pd.DataFrame(stock,index=hs300.index)
    signal = pd.DataFrame(u'其余股票',index=stock.index,columns=pn_data['volume'].columns)
    for date in stock.index:
        hs300_td = hs300.ix[date]
        zz500_td = zz500.ix[date]
        stock_td = stock.ix[date]
        signal.ix[date][set(hs300_td)&set(signal.columns)] = u'沪深300'
        signal.ix[date][set(zz500_td)&set(signal.columns)] = u'中证500'

    signal = format_factor(signal)

    groupby = dict()
    groupby['cap'] = signal


    #set the period
    periods = [1, 2, 4, 5, 10, 20]


    path = ''


    # caculate alpha_data and analyse
    for alpha_name in alpha_function:
        try:
            alpha = eval('gtja.%s()' % alpha_name)

            handle_factor(alpha, gtja.close.copy(), groupby, periods, path)

        except Exception as e:
            print e








  







