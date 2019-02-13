# -*- coding: UTF-8 -*-

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import data

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
        y.append(0.5*math.sin(x))
    return y

def buildWaveSequence(sequenceLength, t, d = 0):
    move = sequenceLength - 1
    start = 0 + d
    y = []
    for i in range(sequenceLength):
        x = start + i * (t * 2 * math.pi) / move
        y.append(0.5*math.sin(x)+0.28*math.sin(3*x))
    return y

# 序列长度
point = 50
sequenceLength = 100
numberOfLSTMUnit = 7
inputSequence = buildWaveSequence(sequenceLength, 4)

(x, y) = data.sequenceFormat(inputSequence, point)

model = createLSTMModel(point, 1, numberOfLSTMUnit, False)
model.compile(optimizer = tf.keras.optimizers.Adam(0.002), loss = "MSE")
model.fit(x, y, 10, 200)

predictSequence = data.predictSequence(model, inputSequence, point, sequenceLength)

plt.plot(range(sequenceLength), inputSequence, 'r')
plt.plot(range(sequenceLength), predictSequence, 'g')
plt.show()
