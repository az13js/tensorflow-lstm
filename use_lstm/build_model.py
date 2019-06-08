# -*- coding: UTF-8 -*-

import tensorflow as tf

def create_model(length = 10, columns = 20):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(
        input_shape = (length, columns),
        units = columns,
        activation = "tanh",
        recurrent_activation = "sigmoid",
        use_bias = True,
        kernel_initializer = "glorot_uniform",
        recurrent_initializer = "orthogonal",
        bias_initializer = "zeros",
        unit_forget_bias = True,
        kernel_regularizer = None,
        recurrent_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        recurrent_constraint = None,
        bias_constraint = None,
        dropout = 0.0,
        recurrent_dropout = 0.0,
        implementation = 1,
        return_sequences = True,
        return_state = False,
        go_backwards = False,
        stateful = False,
        unroll = False
    ))
    model.add(tf.keras.layers.LSTM(
        units = 1,
        activation = "tanh",
        recurrent_activation = "sigmoid",
        use_bias = True,
        kernel_initializer = "glorot_uniform",
        recurrent_initializer = "orthogonal",
        bias_initializer = "zeros",
        unit_forget_bias = True,
        kernel_regularizer = None,
        recurrent_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        recurrent_constraint = None,
        bias_constraint = None,
        dropout = 0.0,
        recurrent_dropout = 0.0,
        implementation = 1,
        return_sequences = False, # False 用后一个输出来作为预测值就够了
        return_state = False,
        go_backwards = False,
        stateful = False,
        unroll = False
    ))
    return model

if __name__ == "__main__":

    model = create_model()
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    model.save("model.h5", overwrite=True, include_optimizer=True)

    # 打印模型
    model.summary()
