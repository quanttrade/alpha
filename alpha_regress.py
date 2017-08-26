from barra_factor import *
from neutralize import *
from scipy import stats
from cvxpy import *


def alpha_delete(alpha_corr, alpha_IR, corr_limit, ir_limit, period):
    alpha_corr_abs = alpha_corr.abs()
    alpha_list = alpha_corr_abs.columns
    alpha_IR_period = alpha_IR.ix[period][['Ann. IR', 'alpha_name']].set_index('alpha_name')['Ann. IR'].abs()

    for alpha_i in alpha_list:
        for alpha_j in alpha_list:
            if alpha_i == alpha_j:
                continue

            if alpha_i in alpha_corr_abs.columns and alpha_j in alpha_corr_abs.columns:

                if alpha_corr_abs.loc[alpha_i, alpha_j] > corr_limit:
                    to_delete = alpha_IR_period.ix[[alpha_i, alpha_j]].argmin()
                    alpha_corr_abs = alpha_corr_abs.drop(to_delete, axis=1)
                    alpha_corr_abs = alpha_corr_abs.drop(to_delete, axis=0)

    IR = alpha_IR_period.ix[alpha_corr_abs.index]
    IR = IR[IR > ir_limit].dropna()
    return IR


def get_alpha_df(alpha_list, path):

    alpha_df = pd.DataFrame()
    for alpha in alpha_list:
        alpha_path = os.path.join(path, alpha, 'neutralize_factor.h5')
        df = pd.read_hdf(alpha_path, 'table')
        alpha_df[alpha] = df.stack()
    return alpha_df



def compute_forward_pure_returns(barra_factor, price_data, periods=(1, 5, 10)):
    """
    Finds the N period forward returns (as percent change) for each asset provided.

    Parameters
    ----------
    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods : sequence[int]
        periods to compute forward returns on.


    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    forward_returns = pd.DataFrame()

    for period in periods:
        factor_returns, rsquare, resid_returns = caculate_factor_returns(barra_factor, price_data, period)
        forward_returns[period] = resid_returns.stack()


    forward_returns.index = forward_returns.index.rename(['date', 'asset'])

    return forward_returns

def get_forward_returns_columns(columns):
    return columns[columns.astype('str').str.isdigit()]


def caculate_adjusted_IC(factor, forward_returns):

    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.
    forward_returns = compute_forward_returns(prices)
    forward_returns['factor'] = factor_data

    Parameters
    ----------
    factor : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for each period,
        The factor quantile/bin that factor value belongs too, and (optionally) the group the

    prices : pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """
    factor_data = forward_returns.copy()
    factor_data['factor'] = factor

    def src_ic(group):
        f = group['factor']
        _ic = group[get_forward_returns_columns(factor_data.columns)].apply(lambda x: stats.pearsonr(x, f)[0])
        return _ic

    grouper = [factor_data.index.get_level_values('date')]
    ic = factor_data.groupby(grouper).apply(src_ic)
    ic.columns = pd.Int64Index(ic.columns)

    return ic


def factor_information_coefficient_statistics(ic):
        ic_summary_table = pd.DataFrame()
        ic_summary_table["IC Mean"] = ic.mean()
        ic_summary_table["IC Std."] = ic.std()
        t_stat, p_value = stats.ttest_1samp(ic, 0)
        ic_summary_table["t-stat(IC)"] = t_stat
        ic_summary_table["p-value(IC)"] = p_value
        ic_summary_table["IC Skew"] = stats.skew(ic)
        ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic)
        ic_summary_table["Ann. IR"] = (
            ic.mean() / ic.std()) * np.sqrt(252)


def stock_weight(bench_weight, risk_factor, volume_t, returns_t, alpha_factor, alpha_returns, upper_bound, TC, w_old):
    """
    return the potofolio weight which make the returns best

    Parameters
    bench_weight: Seriees
    the cap_weight of the benchmark

    risk_factor: DataFrame
    barra_factor including risk factors and industry factors

    volume_t :Series
    the volume if today

    returns_t: Series
    the retuens of today

    alpha_factor: DataFrame
    the loading the alpha_factor

    factor_returns: DataFrame

    upper_bound: float
    the upper bound of stock weight

    TC: float
    Turnover fee

    w_old: array
    the stock weight last day
    """
    # first caculate the bench loading
    bench_risk_loading = risk_factor.ix[bench_weight.index].values.T.dot(bench_weight.values)

    #restrict the stock tradeable
    risk_factor = risk_factor[volume_t > 0]
    risk_factor = risk_factor[returns_t.abs()] < 0.1
    alpha_factor = alpha_factor[risk_factor.index]

    #defne the weight vector to solve
    N = risk_factor.shape[0]
    w = Variable(N, 1)

    #caculate the potofolio loading
    potofolio_risk_loading = risk_factor.values.T * w
    alpha_loading = alpha_factor.values.T * w
    untrade_weight = 0.0
    #the weight of last day
    if isinstance(w_old, float):
        w_last = 0.0

    else:
        w_last = pd.Series(np.repeat(0.0, N), index=risk_factor.index)
        stock_t = list(set(w_last.index) & set(w_old.index))
        w_last.ix[stock_t] = w_old.ix[stock_t]
        untrade_stock = list(set(w_old.index) - set(w_last.index))
        for stock in untrade_stock:
            untrade_weight += w_old[stock]

    #define the object to solve
    ret = alpha_loading * alpha_returns - TC * abs(w - w_last)

    constraints = [w >= 0, w <= upper_bound, sum_entries(w) == 1, abs(potofolio_risk_loading - bench_risk_loading) < 0.01]
    prob = Problem(Maximize(ret), constraints)
    Maximize_value = prob.solve()
    weight = array(w.value)
    weight = pd.Series([weight[i][0] for i in range(len(weight))], index=risk_factor.index) * (1 - untrade_weight)

    if untrade_stock:
        for stock in untrade_stock:
            weight[stock] = w_old[stock]
    return weight




def alpha_model_backtest(barra_factor, alpha_facotr, alpha_returns, benchmark_component, price_data, upper_bound, TC):
    """
    caculate the alpha models
    """

    volume = price_data['volume']
    returns = price_data['adjclose'].pct_change(period).shift(-period)
    cap = price_data['close'] * price_data['total_shares']
    barra_factor.index = barra_factor.index.rename(['date', 'asset'])

    tradedate = barra_factor.index.get_level_values('date')
    tradedate = list(set(tradedate))
    tradedate.sort()
    tradedate = pd.Series(tradedate)
    reblance_day = tradedate.ix[tradedate.index % 2 == 0]

    weight_dict = dict()
    now_weight = pd.Series()

    Lambda = np.power(0.5, 1.0 / 60.0)
    decay_weight = np.array([Lambda ** (249 - i) for i in range(i)])

    for date in tradedate:
        if date not in reblance_day:
            weight_dict[date] = now_weight
        else:
            cap_t = cap.ix[date]
            returns_t = returns.ix[date]
            volume_t = volume.ix[date]

            risk_factor = barra_factor.ix[date]
            benchmark_component_date = benchmark_component.ix[date]
            bench_weight = cap_t.ix[risk_factor.index].ix[benchmark_component_date].dropna()
            bench_weight = bench_weight / bench_weight.sum()

            if now_weight.empty:
                w_old = 0.0

            else:
                w_old = now_weight.copy()


            alpha_returns_date = np.average(alpha_returns.ix[:date][-250:], weights=decay_weight, axis=1)

            reblance_weight = stock_weight(bench_weight=bench_weight, risk_factor=risk_factor,
                                           volume_t=volume_t, returns_t=returns,
                                           alpha_factor=alpha_facotr.ix[date], alpha_returns=alpha_returns_date
                                           upper_bound=upper_bound, TC=TC, w_old=w_old)

            now_weight = reblance_weight

            weight_dict[date] = now_weight























if __name__ == '__main__':
    gtja_path = 'E:\gtja_alpha'
    alpha_dir = os.listdir(gtja_path)
    barra_factor = pd.read_hdf('D:data/daily_data/barra_factor_cap.h5','barra_factor')
    price_data = pd.read_hdf('D:\data\daily_data\\price_data.h5','table')
    periods = [1, 2, 3, 4, 5, 10, 20]
    forward_returns = compute_forward_pure_returns(barra_factor, price_data, periods)
    forward_returns.to_hdf('D:\data\daily_data\\forward_returns.h5', 'table')

    for alpha_name in alpha_dir:
        print alpha_name
        try:
            path = os.path.join(gtja_path, alpha_name, )
            neutralized_factor = pd.read_hdf(path + '\\neutralize_factor.h5', 'table')
            adjusted_ic = caculate_adjusted_IC(neutralized_factor.stack(), forward_returns)
            ic_summary_table = factor_information_coefficient_statistics(adjusted_ic)
            ic_summary_table.to_excel(path + '\\adjusted_ic_summary.xlsx')

        except Exception as e:
            print e
            continue