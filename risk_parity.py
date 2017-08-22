import pandas as pd
import numpy as np
from cvxpy import *





def get_smart_weight(cov_mat, method='min variance', wts_adjusted=False):
    '''
    功能：输入协方差矩阵，得到不同优化方法下的权重配置
    输入：
        cov_mat  pd.DataFrame,协方差矩阵，index和column均为资产名称
        method  优化方法，可选的有min variance、risk parity、max diversification、equal weight
    输出：
        pd.Series  index为资产名，values为weight
    PS:
        依赖scipy package
    '''

    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError('cov_mat should be pandas DataFrame！')

    omega = np.matrix(cov_mat.values)  # 协方差矩阵

    # 定义目标函数
    def fun1(x):
        return np.matrix(x) * omega * np.matrix(x).T

    def fun2(x):
        tmp = (omega * np.matrix(x).T).A1
        risk = x * tmp
        delta_risk = [sum((i - risk)**2) for i in risk]
        return sum(delta_risk)

    def fun3(x):
        den = x * omega.diagonal().T
        num = np.sqrt(np.matrix(x) * omega * np.matrix(x).T)
        return num/den

    # 初始值 + 约束条件
    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple((0,None) for x in x0)
    cons = ({'type':'eq', 'fun': lambda x: sum(x) - 1})
    options={'disp':False, 'maxiter':1000, 'ftol':1e-20}

    if method == 'min variance':
        res = minimize(fun1, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'risk parity':
        res = minimize(fun2, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'max diversification':
        res = minimize(fun3, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'equal weight':
        return pd.Series(index=cov_mat.index, data=1.0 / cov_mat.shape[0])
    else:
        raise ValueError('method should be min variance/risk parity/max diversification/equal weight！！！')

    # 权重调整
    if res['success'] == False:
        # print res['message']
        pass
    wts = pd.Series(index=cov_mat.index, data=res['x'])
    if wts_adjusted == True:
        wts = wts[wts >= 0.0001]
        return wts / wts.sum() * 1.0
    elif wts_adjusted == False:
        return wts
    else:
        raise ValueError('wts_adjusted should be True/False！')



def risk_parity(df, epsilon):
    n = df.shape[1]
    Sigma = df.cov().values
    y0 = np.random.random(n + 1)
    y0[:-1] = 1.0 / (Sigma.diagonal() ** 0.5)
    y0[-1] = 0.5
    flag = True
    y_old = y0
    while flag:
        y_new = np.array(y_old - J_inverse(y_old, Sigma).dot(F(y_old, Sigma)))[0]
        print y_new
        if np.sum((y_new - y_old)**2) < epsilon:
            flag = False
        y_old = y_new
    return y_old



def F(y, Sigma):
    x = y[: -1]
    Lambda = y[-1]
    n = Sigma.shape[0]
    a = np.zeros(n + 1)
    a[: -1] = Sigma.dot(x) - Lambda * 1.0 / x
    a[-1] = x.sum() - 1
    return a


def J_inverse(y, Sigma):
    x = y[:-1]
    Lambda = y[-1]
    n = Sigma.shape[0]
    a = np.matrix(np.zeros([n +1, n + 1]))
    a[:-1,:-1] = Sigma + Lambda * np.diag(1.0 / x ** 2) - 1.0 / x
    a[-1, :-1] = np.zeros(n) + 1
    a[:,-1][:-1] = (- 1.0 / x).reshape(-1, 1)
    a[-1][-1] = 0
    return a.I



def func(x, Sigma):
    n = Sigma.shape[0]
    f = 0.0
    for i in range(n):
        for j in range(n):
            TRCi = x[i] *((Sigma.dot(x))[i])
            TRCj = x[j] *((Sigma.dot(x))[j])
            f += (TRCi - TRCj) ** 2
    return f
    

def func(x, Sigma):
    n = Sigma.shape[0]
    f = 0.0
    for i in range(n):
        for j in range(n):
            TRCi = x[i] *((Sigma.dot(x))[i])
            TRCj = x[j] *((Sigma.dot(x))[j])
            f += (TRCi - TRCj) ** 2
