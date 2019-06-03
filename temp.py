# -*- coding: UTF-8 -*-

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    # 这个必须是一个整数
    input_shape = 20

    # 原始实现是用 Keras 框架做的，这里除了特殊指定之外，其余参数用 Keras 默认的参数配置
    model = tf.keras.models.Sequential()
    # 输入层
    model.add(tf.keras.layers.InputLayer(input_shape=(1, input_shape)))
    # 第一层 LSTM
    model.add(tf.keras.layers.LSTM(
        units=5, # 指定为5
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=tf.keras.regularizers.l2(l=0), # 指定为0
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0.003), # 指定为0.003
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.2, # 指定为0.2
        recurrent_dropout=0.2, # 指定为0.2
        implementation=1,
        return_sequences=True, # 指定为False
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False
    ))
    # Dense 层
    model.add(tf.keras.layers.Dense(
        units=5, # 指定为5
        activation='sigmoid', # 指定为sigmoid
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0.005), # 指定为0.005
        kernel_constraint=None,
        bias_constraint=None
    ))
    # 第二层 LSTM
    model.add(tf.keras.layers.LSTM(
        units=2, # 指定为2
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros',
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=tf.keras.regularizers.l2(l=0.001), # 指定为0.001
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0.01), # 指定为0.003
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.2, # 论文指定为0.2
        recurrent_dropout=0.2, # 论文指定为0.2
        implementation=1,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False
    ))
    # Dense 层
    model.add(tf.keras.layers.Dense(
        units=1, # 指定为1
        activation='sigmoid', # 指定为sigmoid
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=tf.keras.regularizers.l2(l=0.001), # 指定为0.005
        kernel_constraint=None,
        bias_constraint=None
    ))
    return model

if __name__ == "__main__":

    model = create_model()
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

    # 打印模型
    model.summary()

    seq_len = 20
    output_size = 1
    total = 6000
    test_inputs = []
    for i in range(total):
        input = []
        for j in range(seq_len):
            input.append(0.1 * j)
        test_inputs.append([input])
    test_outputs = []
    for i in range(total):
        output = []
        for j in range(output_size):
            output.append(math.sin(0.1 * (j + 1)))
        test_outputs.append(output)

    input_data = np.array(test_inputs)
    output_data = np.array(test_outputs)
    model.fit(input_data, output_data, 4000, 2000)


    #input_data = np.array([[[0.1,0.2,0.3]], [[0.1,0.2,0.3]], [[0.1,0.2,0.3]]])
    #output_data = np.array([[0.1], [0.1], [0.1]])
    #model.fit(input_data, output_data, 100, 2000)
