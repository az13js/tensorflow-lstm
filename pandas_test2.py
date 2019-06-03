# -*- coding: UTF-8 -*-

# 使用 Pandas 将斐波那契数列保存到 CSV 文件，并且带有表格头部
# 然后读取文件并将文件内容转换为数组
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

import pandas as pd

# 产生文件
data = [[1, 1], [2, 1]]
for i in range(10):
    data.append([i + 3, data[i + 1][1] + data[i][1]])
data_frame = pd.DataFrame(data, columns=['x', 'y'])
data_frame.to_csv('test.csv')

# 读取数据并转换为数组
new_data_frame = pd.read_csv('test.csv')
data = new_data_frame.get_values()
print(data)

# 可以用 len() 函数计算 DataFrame 对象数据的行数（不含标题）
print('lines:')
print(len(new_data_frame))

# 从索引 0 开始拿10行，每行取第 2 列。从 0 开始计数
print('type:')
print(type(new_data_frame.iloc[0:10, 2]))
print('data:')
print(new_data_frame.iloc[0:10, 2])
