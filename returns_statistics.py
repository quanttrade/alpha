def maxdrawdown(net_value):
    draw_down = pd.Series([(net_value.iloc[i] - net_value[i:].min()) / net_value.iloc[i] for i in range(len(net_value) - 1)], index=net_value.index[:-1])

    maxdrawdown_value = draw_down.max()
    """
    maxdrawdown_index_start = draw_down.argmax()
    maxdrawdown_index_end = net_value[maxdrawdown_index_start:].argmin()
    """
    return maxdrawdown_value



def volatility(net_value):
    return net_value.pct_change().std() * np.sqrt(252.0)


def returns_(netvalue):
    return netvalue.iloc[-1] / netvalue.iloc[0] - 1

def statistics(netvalue):
    netvalue = netvalue.dropna()
    return net_value.groupby(lambda x: x.year).agg([maxdrawdown, volatility, returns_])
