# -*- coding: UTF-8 -*-

import tensorflow as tf

def create_model(input_shape = 20):
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
    model.save("model.h5", overwrite=True, include_optimizer=True)

    # 打印模型
    model.summary()

