# -*- coding: UTF-8 -*-

# 构造假数据

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def show_build_data(file):
    data_frame = pd.read_csv(file)
    (length, columns) = data_frame.shape
    for i in range(1, columns):
        plt.plot(range(length), np.array(data_frame.iloc[0:length, i]))
    plt.show()

def diff_anyliz(file_name, save_file, column = 20):
    data_frame = pd.read_csv(file_name)
    (length, columns) = data_frame.shape
    data = [np.log(np.array(data_frame.iloc[0:length, 1]))]
    for i in range(column):
        data.append(np.diff(data[i]))
    result = []
    for i in range(len(data[0])):
        lines = []
        for j in range(len(data)):
            if i - j < 0:
                lines.append(0.0)
            else:
                lines.append(data[j][i - j])
        result.append(lines)
    pd.DataFrame(result).to_csv(save_file)

if __name__ == "__main__":
    build('data.csv', 100, 1)
    diff_anyliz('data.csv', 'al.csv', 20)
    # show_build_data('al.csv')
