# -*- coding: UTF-8 -*-

# 构造假数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# 函数作用：构建带噪声正弦序列
# 参数和参数含义：
# file 保存的文件名。函数调用生成的数据会保存并覆盖此文件。
# lines 数据行数，不含表头。默认是3行
def build(file, lines = 3):
    data = []
    for i in range(lines - 1):
        data.append([math.sin(2 * math.pi * 0.01 * i) + 0.1 * np.random.normal(0, 0.5)])
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
    build('sin.csv', 1000) # 构造1000行数据
    show_build_data('sin.csv') # 读取并显示生成的数据。
