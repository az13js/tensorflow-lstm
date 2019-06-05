# -*- coding: UTF-8 -*-

# 时序序列切割

import numpy as np
import pandas as pd

def create_data():
    data = [[1, 1], [2, 1]]
    for i in range(10):
        data.append([i + 3, data[i + 1][1] + data[i][1]])
    data_frame = pd.DataFrame(data, columns=['x', 'y'])
    data_frame.to_csv('test.csv')

def process(csv_file, input_data_file, output_data_file, input_length, output_length, column = 0):
    data_frame = pd.read_csv(csv_file)
    data_line_total = len(data_frame)
    if data_line_total < input_length + output_length:
        return False
    input_row_data = []
    output_row_data = []
    for i in range(data_line_total - input_length - output_length + 1):
        input_sequence = np.array(data_frame.iloc[i : i + input_length, column])
        output_sequence = np.array(data_frame.iloc[i + input_length : i + input_length + output_length, column])
        input_row_data.append(input_sequence)
        output_row_data.append(output_sequence)
    pd.DataFrame(input_row_data).to_csv(input_data_file)
    pd.DataFrame(output_row_data).to_csv(output_data_file)
    return True

if __name__ == "__main__":
    create_data()
    process('test.csv', 'data_input.csv', 'data_output.csv', 8, 2, 2)
