# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import statsmodels.api as sm

def load_data(path):
    adjClose = pd.read_csv(path+ '\\' + 'adjClose.csv',index_col=0)
    adjHigh = pd.read_csv(path+ '\\' + 'adjHigh.csv',index_col=0)
    adjLow = pd.read_csv(path+ '\\' + 'adjLow.csv',index_col=0)
    adjOpen = pd.read_csv(path+ '\\' + 'adjOpen.csv',index_col=0)
    volume = pd.read_csv(path+ '\\' + 'volume.csv',index_col=0)
    pn_data={'adjOpen' :adjOpen,'adjHigh': adjHigh,'adjLow': adjLow,'adjClose': adjClose,'volume':volume}
    return pn_data


def winsorize_series(se):
    q = se.quantile([0.02, 0.98])
    if isinstance(q, pd.Series) and len(q) == 2:
        se[se < q.iloc[0]] = q.iloc[0]
        se[se > q.iloc[1]] = q.iloc[1]
    return se


def winsorize(factor):
    return factor.apply(winsorize_series, axis=1)


def standarlize(factor):
    factor = factor.dropna(how='all')
    factor
    factor_std = ((factor.T - factor.mean(axis=1)) / factor.std(axis=1)).T
    return factor_std

def quick_backtest(factor, pctchange, group_num, period):
    stock_group = group_by_factor(factor, group_num)
    pctchange = pctchange.ix[factor.index]
    res = group_returns_result(stock_group, pctchange, period)
    return res


def group_by_factor(factor, group_num):
    stock_group = dict()
    for i in range(1, group_num + 1):
        stock_group[i] = []
    for line in range(factor.shape[0]):
        cross_data = factor.iloc[line].copy()
        cross_data = cross_data.dropna().sort_values()
        interval = len(cross_data) / group_num
        for quantile in range(1, group_num + 1):
            stock_group[quantile].append(
                cross_data[(quantile - 1) *
                           interval:quantile *
                           interval].index)
    return stock_group


def group_returns_result(stock_group, pctchange, period):
    net_value = dict()
    for group_name in stock_group.keys():
        group = stock_group[group_name]
        group_returns = pd.Series([pctchange.iloc[i][group[i - period]].mean()
                                   for i in range(period, len(pctchange))]) + 1
        net_value[group_name] = group_returns.cumprod()
    res = pd.DataFrame(net_value)
    res.index = pctchange.index[period:]
    return res


def group_mean_return(factor, pctchange, group_num, period):
    stock_group = group_by_factor(factor, group_num)
    group_return = dict()
    pctchange = pctchange.ix[factor.index]
    for key in stock_group.keys():
        group_return[key] = pd.Series([pctchange.iloc[i][stock_group[key][i - period]].mean()
                                       for i in range(period, len(pctchange))]).mean()
    return pd.Series(group_return)



def group_backtest(factor, close, volume, group_num, quantile, fee, period):
    pct_chg = close.pct_change()
    stockpool = pd.Series(np.zeros(factor.shape[1]), index=factor.columns)
    cash = 1.0
    net_value = []
    for i in range(1, factor.shape[0], period):
        date = factor.index[i]
        factor_today = factor.ix[factor.index[i - 1]].sort_values().dropna()
        close_today = close.ix[date]
        pct_chg_today = pct_chg.ix[date]
        vol_today = volume.ix[date]
        inteval_len = factor_today.shape[0] / group_num
        tobuy = factor_today[
            (quantile - 1) * inteval_len:quantile * inteval_len].index
        tosell = stockpool[stockpool > 0].index
        first_sell = list(set(tosell) - set(tobuy))
        for stock in first_sell:
            if pct_chg_today[stock] > -0.099 and vol_today[stock] > 0:
                cash += close_today[stock] * stockpool[stock] * (1 - fee)
                stockpool[stock] = 0.0
        last_buy = list(set(tobuy) - set(tosell))
        buy_num = len(last_buy)
        if buy_num > 0:
            per_money = cash / (buy_num + 0.0)
        for stock in last_buy:
            if pct_chg_today[stock] < 0.99 and vol_today[stock] > 0:
                stockpool[stock] += per_money / close_today[stock] * (1 - fee)
                cash -= per_money

        pool = stockpool[stockpool > 0]
        net_value.append((pool * close_today[pool.index]).sum() + cash)
    return pd.Series(
        net_value,
        index=factor.index[
            range(
                1,
                factor.shape[0],
                period)])


def quantile_mkt_values(signal_df, mkt_df):
    n_quantile = 10
    # 统计十分位数
    cols_mean = [i + 1 for i in range(n_quantile)]
    cols = cols_mean

    mkt_value_means = pd.DataFrame(index=signal_df.index, columns=cols)

    # 计算分组的市值分位数平均值
    for dt in mkt_value_means.index:
        if dt not in mkt_df.index:
            continue
        qt_mean_results = []

        tmp_factor = signal_df.ix[dt].dropna()
        tmp_mkt_value = mkt_df.ix[dt].dropna()
        tmp_mkt_value = tmp_mkt_value.rank() / len(tmp_mkt_value)

        pct_quantiles = 1.0 / n_quantile
        for i in range(n_quantile):
            down = tmp_factor.quantile(
                pct_quantiles * i)
            up = tmp_factor.quantile(pct_quantiles * (i + 1))
            i_quantile_index = tmp_factor[
                (tmp_factor <= up) & (
                    tmp_factor >= down)].index
            mean_tmp = tmp_mkt_value[i_quantile_index].mean()
            qt_mean_results.append(mean_tmp)
        mkt_value_means.ix[dt] = qt_mean_results
    mkt_value_means.dropna(inplace=True)
    return mkt_value_means.mean()


def t_value(factor, pctchange, period):
    tvalues = []
    rsquares = []
    new_factor = factor.copy().dropna(how='all')
    pctchange_copy = pctchange.ix[new_factor.index]
    for i in range(new_factor.shape[0] - period):
        factor_value = new_factor.iloc[i].dropna()
        pct_chg = pctchange_copy.iloc[
            i +
            period].ix[
            factor_value.index].dropna()
        factor_value = factor_value[pct_chg.index]
        results = sm.OLS(pct_chg, factor_value).fit()
        tvalue = results.tvalues[0]
        rsquare = results.rsquared
        tvalues.append(tvalue)
        rsquares.append(rsquare)
    return pd.DataFrame({'tvalue': tvalues, 'rsquare': rsquares},
                        index=new_factor.index[:-period])


def tvalue_sta(tvalues):
    positive = tvalues[tvalues > 2].shape[0]
    negtive = tvalues[tvalues < -2].shape[0]
    total = tvalues.shape[0]
    return [(positive + negtive + 0.0) / total, (positive + 0.0) / negtive]


def returns_sta(net_value):
    grouped = net_value.groupby(lambda x: x.split('-')[0])
    return grouped.apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)


def cap_neutral(factor, mkt_value):
    new_factor = factor.copy().dropna(how='all')

    for i in range(new_factor.shape[0]):
        a = new_factor.iloc[i].dropna()
        mkt = mkt_value.iloc[i].ix[a.index].dropna()
        a = a[mkt.index]
        resduies = sm.OLS(a, mkt).fit().resid
        new_factor.iloc[i].ix[a.index] = resduies

    return new_factor


def alpha_group_return(alpha_list, pctchange, group_num, period, func):
    returns_list = []

    for alpha in alpha_list:
        returns = group_mean_return(alpha, pctchange, group_num, period)
        returns_list.append(returns)

    index = map(lambda x: 'group'+str(x), range(1,period + 1))
    returns_df = pd.DataFrame(returns_list).T
    returns_df.index = index
    returns.columns = func
    return returns_df


def alpha_real_return(alpha_list, func, close, volume, group_num, num, fee, period):
    group_backtest_dic = dict()
    
    for alpha in alpha_list:
        returns = group_backtest(alpha, close, volume, group_num, num, fee, period)
        group_backtest_dic[alpha]= returns
        
    return group_backtest_dic


def alpha_year_return(real_return_df, benchmark_return, func):
    alpha_excess_return = dict()

    for alpha in real_return_df.keys():
        year_return = returns_sta(real_return_df[alpha])
        excess_return = year_return - benchmark_return
        alpha_excess_return[alpha] = excess_return

    df = pd.DataFrame(alpha_excess_return).T
    df.columns = func
    return df


def alpha_tvalue_rsquare(alpha_list, pctchange, period, func):
    tvalues = []
    rsquare = []
    for alpha in alpha_list:
        df = t_value(alpha, pctchange, period)
        tvalues.append(df.tvalues)
        rsquare.append(df.rsquare)
    tvalues,rsquare =  pd.DataFrame(tvalues).T, pd.DataFrame(rsquare).T
    tvalues.columns = func
    rsquare.columns = func
    return tvalues,rsquare


def obvious(tvalues):
    total_ratio_list = []
    compare_ratio_list = []

    for alpha in tvalues.columns:
        total_ratio, compare_ratio = tvalue_sta(tvalues[alpha])
        total_ratio_list.append(total_ratio)
        compare_ratio_list.append(compare_ratio)
    
    return pd.DataFrame({'total_ratio':total_ratio_list,'compare_ratio':compare_ratio_list}, index=tvalues.columns)







































