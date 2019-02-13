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
