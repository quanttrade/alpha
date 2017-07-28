import alphalens
import pandas as pd
from scipy import stats
import numpy as np



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
                                                                                periods)

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


    writer = pd.ExcelWriter(path)
    factor.to_excel(writer, 'prime_factor')
    factor_data_standard.to_excel(writer, 'factor_data_standard')
    quantile_returns_mean_standard.to_excel(writer, 'quantile_returns_mean_standard')
    ic_standard.to_excel(writer, 'ic_standard')
    ic_summary_table.to_csv(writer, 'ic_summary_table')




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
                                                                               periods,
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
        factor_data_key.to_excel(writer, 'factor_data_' + key)
        factor_returns.to_excel(writer, 'factor_returns_' + key)
        quantile_returns_mean_key.to_excel(writer, 'quantile_returns_mean_' + key)
        ic.to_excel(writer, 'ic_' + key)
        ic_table.to_excel(writer, 'ic_table_' + key)
        quantile_turnover_mean.to_excel(writer, 'quantile_turnover_mean_' + key)







