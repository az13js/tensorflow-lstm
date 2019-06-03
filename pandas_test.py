# -*- coding: UTF-8 -*-

# 使用 Pandas 将斐波那契数列保存到 CSV 文件，并且带有表格头部
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

import pandas as pd

# 第一个数字是1，第二个数字是1
# 有多少行就有多少个元素（可以有表格头部）
# 元素是1维数组，代表每个单元格的内容
data = [[1, 1], [2, 1]]
for i in range(10):
    data.append([i + 3, data[i + 1][1] + data[i][1]])

data_frame = pd.DataFrame(data, columns=['x', 'y'])
print(data_frame)

data_frame.to_csv('test.csv')
