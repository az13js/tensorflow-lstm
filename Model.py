# -*- coding: UTF-8 -*-

import tensorflow as tf

def softmaxCrossEntropy():
    return tf.losses.softmax_cross_entropy;

def sigmoidCrossEntropy():
    return tf.losses.sigmoid_cross_entropy;

def sparseSoftmaxCrossEntropy():
    return tf.losses.sparse_softmax_cross_entropy;

def getOptimizers():
    return tf.keras.optimizers;

def createLSTMModelForClassification(sequenceLength, dim = 1, LSTMUnitNumber = 2, returnFullOutputSequence = False):
    """"创建一个LSTM模型

    Args:
        sequenceLength: 时间步长度
        dim: 隐含层节点数
        LSTMUnitNumber: LSTM的层数
        returnFullOutputSequence: 是否最后返回整个预测的序列

    Return:
        返回一个Tensorflow的Keras的Sequential层
    """
    if LSTMUnitNumber < 1:
        return False
    if LSTMUnitNumber == 1:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(
            dim,
            #activation = "sigmoid",
            input_shape = (sequenceLength, dim),
            return_sequences = returnFullOutputSequence
        ))
        model.add(tf.keras.layers.Dense(dim, "sigmoid"))
        return model
    # LSTMUnitNumber > 1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(dim, input_shape=(sequenceLength, dim), return_sequences = True))
    for i in range(LSTMUnitNumber - 2):
        model.add(tf.keras.layers.LSTM(dim, return_sequences = True))
    model.add(tf.keras.layers.LSTM(dim, return_sequences = returnFullOutputSequence))
    model.add(tf.keras.layers.Dense(dim, "sigmoid"))
    return model
