# -*- coding: UTF-8 -*-

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

def createLSTMModel(sequenceLength, dim = 1, LSTMUnitNumber = 2, returnFullOutputSequence = False):
    if LSTMUnitNumber < 1:
        return False
    if LSTMUnitNumber == 1:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(
            dim,
            # activation = "sigmoid",
            input_shape = (sequenceLength, dim),
            return_sequences = returnFullOutputSequence
        ))
        return model
    # LSTMUnitNumber > 1
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(dim, input_shape=(sequenceLength, dim), return_sequences = True))
    for i in range(LSTMUnitNumber - 2):
        model.add(tf.keras.layers.LSTM(dim, return_sequences = True))
    model.add(tf.keras.layers.LSTM(dim, return_sequences = returnFullOutputSequence))
    return model

def buildSinSequence(sequenceLength, t, d = 0):
    move = sequenceLength - 1
    start = 0 + d
    y = []
    for i in range(sequenceLength):
        x = start + i * (t * 2 * math.pi) / move
        y.append([0.5*math.sin(x)])
    return y

def buildWaveSequence(sequenceLength, t, d = 0):
    move = sequenceLength - 1
    start = 0 + d
    y = []
    for i in range(sequenceLength):
        x = start + i * (t * 2 * math.pi) / move
        y.append([0.5*math.sin(x)+0.28*math.sin(3*x)])
    return y

# 序列长度
point = 150

x = [buildSinSequence(point, 4)]
y = [buildWaveSequence(point, 4)]

numberOfLSTMUnit = 1

model = createLSTMModel(point, 1, numberOfLSTMUnit, True)
model.compile(optimizer = tf.keras.optimizers.Adam(0.07), loss = "MSE")
model.fit(np.array(x), np.array(y), 1, 300)
y0 = model.predict(np.array(x))
y1 = np.array(y)

# 红色是输入序列，绿色目标序列，蓝色模型预测的序列
plt.plot(range(point),np.reshape(x, (point)), 'r')
plt.plot(range(point),np.reshape(y0, (point)), 'b')
plt.plot(range(point),np.reshape(y1, (point)), 'g')
plt.show()
