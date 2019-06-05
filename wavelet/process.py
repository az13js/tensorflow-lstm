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

if __name__ == "__main__":
    (x, y) = read_data('sin.csv')
    # 分解
    (cA, cD) = pywt.dwt(y, 'haar')
    #plt.plot(range(len(cA)), cA, 'b')
    #plt.plot(range(len(cD)), cD, 'g')
    #plt.show()
    # 重建
    #rebuild = pywt.idwt(cA, cD, 'haar')
    #plt.plot(x, y, 'b')
    #plt.plot(range(len(rebuild)), rebuild, 'g')
    #plt.show()
    # 对分解后的参量进行降噪
    result_cA = pywt.threshold(cA, np.std(cA), mode='soft')
    result_cD = pywt.threshold(cD, np.std(cD), mode='soft')

    # 用降噪后的分量重建，得到经过降噪后的信号
    result_rebuild = pywt.idwt(result_cA, result_cD, 'haar')
    plt.plot(range(len(result_rebuild)), result_rebuild, 'g')
    plt.show()
