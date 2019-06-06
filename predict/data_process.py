# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np

def calculate_init_params(file_name):
    data_frame = pd.read_csv(file_name)
    (length, columns) = data_frame.shape
    column_average = []
    column_beta = []
    for column in range(columns - 1):
        column_data = np.array(data_frame.iloc[0:length, column + 1])
        column_average.append(np.average(column_data))
        column_beta.append(np.max(column_data) - np.min(column_data))
    return (column_average, column_beta)

def process(file_name, average, beta):
    data_frame = pd.read_csv(file_name)
    (length, columns) = data_frame.shape

if __name__ == "__main__":
    (average, beta) = calculate_init_params('al.csv')
    process('al.csv', average, beta)

