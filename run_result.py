from barra_factor import *
from neutralize import *
from alpha_regress import *
from returns_statistics import *
import pandas as pd
import os



if __name__ == '__main__':

    periods = [1, 2, 3, 4, 5, 10, 20]
    alpha_corr = alpha_corr = pd.read_csv('D:data/daily_data/alpha_corr.csv',index_col=0)
    ic_adjust = pd.read_csv('D:data/daily_data/ic_adjust.csv',index_col=0)
    zz500_return = pd.read_hdf('D:data/daily_data/zz500_returns.h5', 'table')
    gtja_path = 'E:\gtja_alpha'
    save_path = 'E:\short_term_alpha'
    barra_factor = pd.read_hdf('D:data/daily_data/barra_factor_cap.h5','barra_factor')
    price_data = pd.read_hdf('D:\data\daily_data\\price_data.h5','table')
    industry_factor = barra_factor.ix[:,10:-1]
    risk_factors = barra_factor.ix[:,:10]
    benchmark_component = pd.read_csv('D:data/daily_data/zz500_component.csv',index_col=0)
    benchmark_component.index = pd.DatetimeIndex(benchmark_component.index)
    weight_bound = 0.02
    risk_loading_bound = 0.01
    industry_loading_bound = 0.05
    TC_list = [0.003, 0.005, 0.04, 0.006, 0.007, 0.008, 0.009, 0.01]
    IR_limit = [3.0, 4.0, 5.0]

    for period in periods:
        print period
        period_path = os.path.join(save_path, 'period_%s' %period)
        if not os.path.exists(period_path):
            os.makedirs(period_path)

        for ir_limit in IR_limit:
            ir_path = os.path.join(period_path, 'IR_%s' % ir_limit)
            if not os.path.exists(ir_path):
                os.makedirs(ir_path)
            IR = alpha_delete(alpha_corr, ic_adjust, 0.4, ir_limit, period)
            alpha_factor = get_alpha_df(list(IR.index), gtja_path)
            total_factor = pd.concat([barra_factor, alpha_factor], axis=1)
            factor_returns, rsquare, resid_returns = caculate_factor_returns(total_factor, price_data, period)
            factor_returns.to_hdf(os.path.join(ir_path, 'factor_returns.h5'), 'table')
            alpha_returns = factor_returns[IR.index]

            for TC in TC_list:
                tc_path = os.path.join(ir_path, 'TC_%s' % TC)
                if not os.path.exists(tc_path):
                    os.makedirs(tc_path)
                    cumulative_return, position = alpha_model_backtest(risk_factors, industry_factor, alpha_factor, alpha_returns,
                                                                       benchmark_component, price_data, weight_bound, risk_loading_bound,
                                                                       industry_loading_bound, TC, period)
                    daily_return = cumulative_return.pct_change().dropna()
                    benchmark_return = zz500_return.ix[daily_return.index]
                    excess_return = daily_return - benchmark_return
                    cumulative_excess_return = (1 + excess_return).cumprod()
                    ret_situation = statistics(cumulative_excess_return)
                    turnover = caculate_turnover(position)
                    turnover = turnover[turnover>0]
                    cumulative_return.to_csv(os.path.join(tc_path, 'cumulative_return.csv'))
                    ret_situation.to_csv(os.path.join(tc_path, 'ret_situation.csv'))
                    position.to_hdf(os.path.join(tc_path, 'position.h5'), 'table')
                    turnover.to_csv(os.path.join(tc_path, 'turnover.csv'))
