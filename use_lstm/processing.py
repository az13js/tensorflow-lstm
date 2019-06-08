# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np

def differential(file_name, save_file, column = 20):
    data_frame = pd.read_csv(file_name)
    (length, columns) = data_frame.shape
    data = [np.array(data_frame.iloc[0:length, 1])]
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

def normalization(file_name, save_file):
    data_frame = pd.read_csv(file_name)
    (length, columns) = data_frame.shape
    column_average = []
    column_beta = []
    for column in range(columns - 1):
        column_data = np.array(data_frame.iloc[0:length, column + 1])
        column_average.append(np.average(column_data))
        #print(column + 1)
        #print(np.average(column_data))
        #print(np.max(column_data) - np.min(column_data))
        column_beta.append(np.max(column_data) - np.min(column_data))
    result = []
    for row in range(length):
        line = []
        for column in range(columns - 1):
            line.append(0.1 * (data_frame.iloc[row, column + 1] - column_average[column]) / column_beta[column])
        result.append(line)
    pd.DataFrame(result).to_csv(save_file)

if __name__ == "__main__":
    differential('data.csv', 'diff.csv', 19)
    print("Differential, OK.")
    normalization("diff.csv", "normalization.csv")
    print("Normalization, OK")
