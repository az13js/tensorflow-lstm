# -*- coding: UTF-8 -*-

# 执行这个文件将会创建 data.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def build(file, lines = 3, columns = 4):
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

def show_build_data(file):
    data_frame = pd.read_csv(file)
    (length, columns) = data_frame.shape
    for i in range(1, columns):
        plt.plot(range(length), np.array(data_frame.iloc[0:length, i]))
    plt.show()

if __name__ == "__main__":
    build('data.csv', 200, 1)
    show_build_data('data.csv')