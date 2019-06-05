# -*- coding: UTF-8 -*-

# 构造假数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 函数作用：构建类似股票价格波动的序列
# 参数和参数含义：
# file 保存的文件名。函数调用生成的数据会保存并覆盖此文件。
# lines 数据行数，不含表头。默认是3行
# columns 数据列数。默认等于1。
def build(file, lines = 3, columns = 1):
    data = []
    line_data = []
    for i in range(columns):
        line_data.append(100.0)
    data.append(line_data)
    for i in range(lines - 1):
        line_data = []
        for j in range(columns):
            if 0 == j:
                line_data.append(float(data[i][j] + 0.1 * np.random.normal(0, 2)))
            else:
                line_data.append(float(data[i][0] + 0.1 * np.random.normal(0, 2)))
        data.append(line_data)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(file)

# 弹出窗口显示指定的文件中的数据。
# 参数：file，指定的数据文件文件名
def show_build_data(file):
    data_frame = pd.read_csv(file)
    (length, columns) = data_frame.shape
    for i in range(1, columns):
        plt.plot(range(length), np.array(data_frame.iloc[0:length, i]))
    plt.show()

if __name__ == "__main__":
    build('data.csv', 1000, 1) # 构造1000行数据，列数为1列
    show_build_data('data.csv') # 读取并显示生成的数据。
