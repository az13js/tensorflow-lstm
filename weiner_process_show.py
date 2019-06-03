# -*- coding: UTF-8 -*-

# 小波变换的降噪演示

import numpy as np
import pywt
import math
import matplotlib.pyplot as plt

length = 600
src = []
for i in range(length):
    src.append(0.0)
data = np.array(src)
# 绿色为原始信号
plt.plot(range(length), data, 'g')

# 对原始信号加噪声
data_noise = np.array(src)
for i in range(length - 1):
    # 维纳过程的模拟
    data_noise[i + 1] = data_noise[i] + 0.5 * np.random.normal(0, 0.4, 1)
# 绘制加噪声的正弦信号为蓝色
plt.plot(range(length), data_noise, 'b')

# 分解
(cA, cD) = pywt.dwt(data_noise, 'haar')

# 对分解后的参量进行降噪
result_cA = pywt.threshold(cA, np.std(cA), mode='soft')
result_cD = pywt.threshold(cD, np.std(cD), mode='soft')

# 用降噪后的分量重建，得到经过降噪后的信号
result_rebuild = pywt.idwt(result_cA, result_cD, 'haar')
# 红色的是降噪后的信号
plt.plot(range(length), result_rebuild, 'r')
plt.show()
