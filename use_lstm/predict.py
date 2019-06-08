# -*- coding: UTF-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def last_data_for_input(file_name, input_length):
    data_frame_all = pd.read_csv(file_name)
    (length, columns) = data_frame_all.shape
    return np.array(data_frame_all.iloc[length - 1 - input_length : length - 1, 1 : columns])

def normalization_params(file_name):
    data_frame = pd.read_csv(file_name)
    (length, columns) = data_frame.shape
    column_average = []
    column_beta = []
    for column in range(columns - 1):
        column_data = np.array(data_frame.iloc[0:length, column + 1])
        column_average.append(np.average(column_data))
        column_beta.append(np.max(column_data) - np.min(column_data))
    return (column_average, column_beta)

def last_diff(file_name, num = 9):
    data_frame_all = pd.read_csv(file_name)
    (length, columns) = data_frame_all.shape
    return np.array(data_frame_all.iloc[length - num : length, 1 : columns])

def predict(length, model_file, diff_file, normalization_file, model_input_length = 10):
    model = tf.keras.models.load_model(model_file)
    (average, beta) = normalization_params(diff_file)
    input_data = last_data_for_input(normalization_file, model_input_length)
    last_diff_data = last_diff(diff_file, model_input_length - 1)
    result = []
    for i in range(length):
        model_output = model.predict(np.asarray([input_data]))
        # 根据平均值和差异计算真实预测值
        predict_value = model_output[0][0] * 10.0 * beta[0] + average[0]
        result.append(predict_value)
        diff_data = []
        diff_data.append(predict_value)
        last_diff_data = last_diff_data[-9:]
        for col in range(len(average) - 1):
            diff_data.append(diff_data[col] - last_diff_data[8][col])
        last_diff_data = np.concatenate((last_diff_data, [diff_data]), axis = 0)
        input_data = last_diff_data.copy()
        for row in range(len(input_data)):
            for col in range(len(input_data[row])):
                input_data[row][col] = 0.1 * (input_data[row][col] - average[col]) / beta[col]
    return result

if __name__ == "__main__":
    predict_data = predict(1000, "finally.h5", "diff.csv", "normalization.csv", 10)
    data_frame = pd.read_csv("data.csv")
    plt.plot(range(len(data_frame)), data_frame.iloc[0:len(data_frame), 1], 'b')
    plt.plot(range(len(data_frame), len(data_frame) + len(predict_data)), predict_data, 'r')
    plt.show()
