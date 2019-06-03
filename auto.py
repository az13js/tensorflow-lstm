# -*- coding: UTF-8 -*-

import tensorflow as tf
import math
import numpy as np

def build():

    input_shape = 55
    shape1 = 40
    shape2 = 30
    shape3 = 20
    shape4 = 30
    shape5 = 40
    shape6 = 55

    input = tf.keras.layers.Input(shape=(1, input_shape))
    dense1 = tf.keras.layers.Dense(
        units=shape1,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0),
        kernel_constraint=None,
        bias_constraint=None
    )(input)
    dense2 = tf.keras.layers.Dense(
        units=shape2,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0),
        kernel_constraint=None,
        bias_constraint=None
    )(dense1)
    dense3 = tf.keras.layers.Dense(
        units=shape3,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0),
        kernel_constraint=None,
        bias_constraint=None
    )(dense2)
    dense4 = tf.keras.layers.Dense(
        units=shape4,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0),
        kernel_constraint=None,
        bias_constraint=None
    )(dense3)
    dense5 = tf.keras.layers.Dense(
        units=shape5,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0),
        kernel_constraint=None,
        bias_constraint=None
    )(dense4)
    dense6 = tf.keras.layers.Dense(
        units=shape6,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0),
        kernel_constraint=None,
        bias_constraint=None
    )(dense5)

    encode_decode = tf.keras.Model(inputs=input, outputs=dense6)
    encode = tf.keras.Model(inputs=input, outputs=dense3)
    return (encode_decode, encode)

if __name__ == "__main__":

    (encode_decode, encode) = build()
    encode_decode.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    encode_decode.summary()
    encode.summary()

    train_data = [];
    for i in range(1000):
        data = []
        for j in range(55):
            data.append(math.sin(j))
        train_data.append([data])
    print('Before train:')
    print(encode.predict(np.array([train_data[0]])))
    encode_decode.fit(np.array(train_data), np.array(train_data), 1000, 1000, verbose=0)
    print('After train:')
    print(encode.predict(np.array([train_data[0]])))
