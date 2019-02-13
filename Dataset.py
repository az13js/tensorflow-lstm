# -*- coding: UTF-8 -*-
import numpy as np

def sequenceFormat(longSequence, length, vectorLength = 1):
    """将一个很长的一维数组转换成可以用于LSTM训练的数据集

    将一个很长的一维列表，按照length个元素组成一个小的列表的方式，截断，将下一
    个元素作为数组，构成训练集。例如：
    [1,2,3,4]
    按照length=2截断后变成
    [1,2]->3
    [2,3]->4
    这样组成2个训练数据。
    这个函数是为了给tf.keras.layer.LSTM做准备的，因此它返回的是3维列表：
    [[[1], [2]]  -> [[3]
     [[2], [3]]] ->  [4]]
    其中，左侧两组是训练输入，右侧两组是训练输出。

    Args:
        longSequence: 一个长列表，结构是1维的。
        length: 截断长度。
        vectorLength: 向量长度，默认是1

    Returns:
        一个元祖，元祖的第一个元素是输入，第二个元素是输出
    """
    inputSequence = []
    predictNumber = []
    dataNumber = len(longSequence) - length
    for i in range(dataNumber):
        inputSequence.append(longSequence[i : (i + length)])
        predictNumber.append(longSequence[i + length])
    return (np.array(inputSequence).reshape((dataNumber, length, vectorLength)), np.array(predictNumber).reshape((dataNumber, vectorLength)))

def predictSequence(model, inputSequence, length, predictLength, dim = 1):
    """根据inputSequence，让model预测predictLength个长度的数据"""
    result = []
    x = inputSequence
    for i in range(predictLength):
        x = x[-length:]
        #print(x)
        y = model.predict(np.array(x).reshape((1,length, dim)))
        max_key = 0
        max_value = y[0][max_key]
        for j in range(len(y[0])):
            if y[0][j] > max_value:
                max_value = y[0][j]
                max_key = j
            y[0][j] = 0
        y[0][max_key] = 1
        x.append(y[0])
        result.append(y[0])
    return result
