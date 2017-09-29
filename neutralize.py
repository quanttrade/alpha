from barra_factor import *
import os
import alphalens
from scipy import stats


if __name__ == '__main__':
    gtja_path = 'E:\multi_factor\prime_factor'
    alpha_dir = os.listdir(gtja_path)
    price_data = pd.read_hdf('E:\multi_factor\price_data\\price_data.h5','table')
    barra_factor = pd.read_hdf('E:/multi_factor/basic_factor/barra_factor_new.h5', 'table')
    cap = price_data['close'] * price_data['total_shares']
    #ic_table_neutralize = pd.DataFrame()


    for alpha_name in alpha_dir:
        try:
            print alpha_name
            print "============================================"
            path = os.path.join(gtja_path, alpha_name)
            prime_factor = pd.read_hdf(path, 'table')
            prime_factor_standard = standardize_cap(winsorize(prime_factor, mad_method), cap)
            neutralized_factor = neutralize(prime_factor_standard.ix['2007-01-31':], barra_factor.ix[:,:-1])
            neutralized_factor = standardize_cap(neutralized_factor.astype(float), cap)
            neutralized_factor.to_hdf(os.path.join('E:\multi_factor\\neutralized_factor', alpha_name), 'table')

            """
            #get ic and other statistics of factor
            neutralized_factor_stack = neutralized_factor.stack()
            prices = price_data['adjclose'].ix[neutralized_factor_stack.index[0][0]:]
            neutralized_factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factor=neutralized_factor_stack,
                                                                                       prices=prices,
                                                                                       periods=(1, 3, 5, 10, 20, 30, 60))
            quantile_returns_mean_standard, quantile_returns_std_standard = alphalens.performance.mean_return_by_quantile(
                neutralized_factor_data)

            ic_standard = alphalens.performance.factor_information_coefficient(
                neutralized_factor_data)

            turnover_periods = alphalens.utils.get_forward_returns_columns(
                    neutralized_factor_data.columns)
            quantile_factor = neutralized_factor_data['factor_quantile']

            quantile_turnover = {p: pd.concat([alphalens.performance.quantile_turnover(
                quantile_factor, q, p) for q in range(1, int(quantile_factor.max()) + 1)],
                axis=1)
                for p in turnover_periods}

            quantile_turnover_mean = pd.Panel(quantile_turnover).mean()

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

            ic_table_neutralize = pd.concat([ic_table_neutralize, ic_summary_table])


            neutralized_factor_data.to_hdf(path + '\\factor_data_neutralize.h5', 'table')
            quantile_returns_mean_standard.to_excel(
                path + '\\quantile_returns_mean_neutralize.xlsx')
            ic_standard.to_excel(path + '\\ic_neutralize.xlsx')
            ic_summary_table.to_excel(path + '\\ic_summary_table_neutralize.xlsx')
            quantile_turnover_mean.to_excel(
                path + '\\quantile_turnover_mean_neutralize.xlsx')
            """

        except Exception as e:
            print e
            continue

    #ic_table_neutralize.to_csv('D:\data\daily_data\\ic_table_neutralize.csv')
