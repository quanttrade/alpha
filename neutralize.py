from barra_factor import *
import os
import alphalens



if __name__ == '__main__':
    gtja_path = 'E:\gtja_alpha'
    alpha_dir = os.listdir(gtja_path)
    barra_factor = pd.read_hdf('')
    price_data = pd.read_hdf('D:\data\daily_data\\price_data.h5','table')
    ic_table_neutralize = pd.DataFrame()


    for alpha_name in alpha_dir:
        path = os.path.join(gtja_path, alpha_name)
        prime_factor = pd.read_hdf(path + '\\prime_factor.h5', 'table')
        prime_factor_standard = standardize_cap(winsorize(prime_factor, mad_method))
        neutralized_factor = neutralize(prime_factor_standard, barra_factor)
        neutralized_factor.to_hdf(path + '\\neutralize_factor.h5', 'table')


        #get ic and other statistics of factor
        neutralized_factor_stack = neutralized_factor.stack()
        prices = price_data['adjclose'].ix[neutralized_factor_stack.index[0][0]:]
        neutralized_factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factor=neutralized_factor_stack,
                                                                                       prices=prices,
                                                                                       periods=(1, 3, 5, 10, 20, 30, 60))
        quantile_returns_mean_standard, quantile_returns_std_standard = alphalens.performance.mean_return_by_quantile(
                neutralized_factor_data)

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


        neutralized_factor_data.to_hdf(path + '\\factor_data_neutralize.h5', 'table')
        quantile_returns_mean_standard.to_excel(
            path + '\\quantile_returns_mean_neutralize.xlsx')
        ic_standard.to_excel(path + '\\ic_neutralize.xlsx')
        ic_summary_table.to_excel(path + '\\ic_summary_table_neutralize.xlsx')
