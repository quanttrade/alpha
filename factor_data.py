import alphalens as al
import pandas as pd



def get_factor_data(factor):
    factor = factor.set_index('tradeDate')
    factor.index = pd.DatetimeIndex(factor.index)
    it = [list(pd.DatetimeIndex(factor.index)), list(factor.columns)]
    index = pd.MultiIndex.from_product(it,names=['tradeDate','secID'])
    factor_data = pd.DataFrame(factor.values.reshape(-1,1), index=index)[0]
    return factor_data



