# Wavelet

## 一维离散小波变换：

    pywt.dwt(data, wavelet)

- data 待处理数据
- wavelet 使用指定的小波基函数

返回值一个2元的Tuple，参数cA和cD，分别为近似分量和细节分量

## 一维离散小波反变换

    pywt.idwt(cA, cD, wavelet)

由近似分量 cA 和细节分量 cD 经小波反变换重构原始信号。

## 小波降噪，阈值函数

    pywt.threshold(data, value, mode='soft', substitute=0)

以 value 值为界限，将 data 中数据设置为 0 。
