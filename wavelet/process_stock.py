# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

def read_data(file):
    data_frame = pd.read_csv(file)
    (length, columns) = data_frame.shape
    x = np.array(data_frame.iloc[0:length, 0])
    y = np.array(data_frame.iloc[0:length, 1])
    return (x, y)

def diff_log(file, save_file):
    (x, y) = read_data(file)
    # wavelet
    (cA, cD) = pywt.dwt(y, 'haar')
    result_cA = pywt.threshold(cA, np.std(cA), mode='soft')
    result_cD = pywt.threshold(cD, np.std(cD), mode='soft')
    rebuild = pywt.idwt(result_cA, result_cD, 'haar')
    pd.DataFrame(100.0 * np.diff(np.log(rebuild))).to_csv(save_file)

if __name__ == "__main__":
    diff_log('data.csv', 'diff.csv')
    data_frame = pd.read_csv('diff.csv')
    length = len(data_frame)
    plt.plot(range(len(np.array(data_frame.iloc[0:length, 1]))), np.array(data_frame.iloc[0:length, 1]), 'g')
    plt.show()