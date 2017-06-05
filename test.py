import pandas as pd
import numpy as np
import statsmodels.api as sm


def simulation(ret_factor, fund_num):
    n,m = ret_factor.shape
    fund_ret_para = np.random.normal(0, 0.01, fund_num)
    fund_beta = []
    for i in range(fund_num):
        beta = np.random.normal(1, 1, [n,m])
        beta = pd.DataFrame(beta, index=ret_factor.index, columns=ret_factor.columns)
        fund_beta.append(beta)
    fund_ret = []
    for i in range(fund_num):
        ret = (ret_factor * fund_beta[i]).sum(axis=1) + pd.Series(np.random.normal(fund_ret_para[0], 0.01, n), index=ret_factor.index)
        fund_ret.append(ret)
    return fund_ret_para, fund_ret


def OLS_alpha(fund_ret, ret_factor):
    alpha = []
    for i in range(len(fund_ret)):
        model = sm.OLS(fund_ret[i], sm.add_constant(ret_factor))
        res = model.fit()
        alpha.append(res.params.const)
    return np.array(alpha)




