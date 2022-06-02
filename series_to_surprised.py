
"""
    对pandas数据 将时序数据转换成可监督数据进行训练
"""

import pandas as pd


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    将时间序列重构为监督学习数据集.
    参数:
    	data: 观测值序列，类型为列表或Numpy数组。
    	n_in: 输入的滞后观测值(X)长度。
    	n_out: 输出观测值(y)的长度。
    	dropnan: 是否丢弃含有NaN值的行，类型为布尔值。
    返回值:
    	经过重组后的Pandas DataFrame序列.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 将列名和数据拼接在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 丢弃含有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


data = series_to_supervised([1,2,3,4,5])
print(data)
