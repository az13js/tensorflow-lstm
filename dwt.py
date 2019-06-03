# -*- coding: UTF-8 -*-

import numpy as np
import pywt

data = np.array([1,2,3,4,5,6,7,8,9,10])

# 分解
(cA, cD) = pywt.dwt(data, 'haar')
print('cA and cD:')
print((cA, cD))

# 重建
rebuild = pywt.idwt(cA, cD, 'haar')
print('Rebuild signal:')
print(rebuild)

# 对分解后的参量进行降噪
result_cA = pywt.threshold(cA, np.std(cA), mode='soft')
result_cD = pywt.threshold(cD, np.std(cD), mode='soft')

# 用降噪后的分量重建，得到经过降噪后的信号
result_rebuild = pywt.idwt(result_cA, result_cD, 'haar')
print('Result:')
print(result_rebuild)
