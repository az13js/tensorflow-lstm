# -*- coding: UTF-8 -*-

# 这份文件的作用是作为一个示例，记录关于 LSTM 函数的各个参数的含义

import tensorflow as tf
import numpy as np

# 这里建立一个简单的模型演示 LSTM 层的特性
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(
    # 这里是作为Sequential模型的第一层所以指定input_shape参数，后面加的层不需要这个
    # 这里的input_shape是两个元素的，第一个代表每个输入的样本序列长度，第二个元素代表
    # 每个序列里面的1个元素具有多少个输入数据。例如，LSTM处理的序列长度为10，每个时间
    # 步即序列的元素是由两个维度组成，那么这个参数设置为(10, 2)
    input_shape = (10, 2),
    # unit 决定了一层里面 LSTM 单元的数量。这些单元是并列的，一个时间步骤里，输入这个
    # 层的信号，会被所有 unit 同时并行处理，形成若 unit 个输出。所以官方文档说，这个
    # unit 参数是这一层输出的维度。如果说上面的input_shape是(10, 2)的话，设个设置为
    # units = 3，那么这一层的输出一般就是(10, 3)，如果返回的h（输出序列）是所有的h
    # 的话。
    units = 3,
    # 激活函数的类型，默认是没有激活函数。这个激活函数决定了每个 unit 每一次计算的输出
    # 范围。如果要模拟原论文的话，应该设置activation = "tanh"。
    activation = "tanh",
    # recurrent activation是作用于 C 的计算，要模拟原文的话需要设置为sigmoid
    recurrent_activation = "sigmoid",
    # 是否增加一个偏置向量。模拟原文的话设置为 True。
    use_bias = True,
    # 用来初始化内核权重矩阵，用于对输入进行线性转换
    kernel_initializer = "glorot_uniform",
    # 回归计算的内核权重矩阵初始化方法，用于给回归状态进行线性转换 orthogonal 是默认
    # 的值
    recurrent_initializer = "orthogonal",
    # 给偏置进行初始化操作的方法，默认值是 zeros
    bias_initializer = "zeros",
    # 如果设置为 True 会给遗忘门增加一个 bias 参数，同时强制设置bias_initializer为
    # zeros
    unit_forget_bias = True,
    # 内核权重归一化的方法，默认为 None
    kernel_regularizer = None,
    # 回归权重矩阵归一化方法，默认None
    recurrent_regularizer = None,
    # 用于偏置矩阵的归一化方法，默认None
    bias_regularizer = None,
    # 给 LSTM 输出的归一化方法，默认None
    activity_regularizer = None,
    # 用于内核参数矩阵的约束函数，默认None
    kernel_constraint = None,
    # 回归参数矩阵的约束函数，默认None
    recurrent_constraint = None,
    # 偏置参数矩阵的约束函数，默认为None
    bias_constraint = None,
    # 使多少比重的神经元输出（unit的输出）激活失效，默认为0，模仿原文为0
    dropout = 0.0,
    # recurrent_dropout是给递归状态 C 设置的Dropout参数
    recurrent_dropout = 0.0,
    # 实现方式。1会将运算分解为若干小矩阵的乘法加法运算，2则相反。这个参数不用情况下
    # 会使得程序具有不同的性能表现。
    implementation = 1,
    # return_sequences 是否返回全部输出的序列。False否，True返回全部输出，框架默认
    # False，模拟原文可以考虑设置为True
    return_sequences = True,
    # 是否返回LSTM的中间状态，框架默认False，模拟原文可以设置为False。这里返回的状态
    # 是最后计算后的状态
    return_state = False,
    # 是否反向处理。设置为诶True则反向处理输入序列以及返回反向的输出。默认为False
    go_backwards = False,
    # 默认为False，如果设置为True，每一个批的索引i代表的样本的最后一个状态量C，将会作
    # 为初始化状态，初始化下一个索引i的批次样本
    stateful = False,
    # 是否展开。把LSTM展开计算会加速回归计算的过程，但是会占用更多内存，建议在小序列上
    # 展开。默认为False。
    unroll = False
))

# 打印调试信息
model.summary()

# 输入批次，包含所有样本，这里就一个样本
input_data_all = [
    # 样本内有10个元素，对应输入序列有10个元素，每个元素里面包含两个子元素，对应每个
    # 时间步 LSTM 单元处理两个输入数据。这个是跟上面构造模型时的input_shape一致的。
    [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]]
]

# 运行上面的模型，得到计算结果
output_data_all = model.predict(np.array(input_data_all, dtype=np.float64))

# 上面的输入只有一个样本，所以输出也只有一个样本，由于指定返回所有输出序列，也就是
# return_sequences = True, 所以这里返回的output_data_all[0]包含了10个元素，对应
# 输入序列10个元素的每个的计算结果，每个结果又是具有3个元素的，对应了上面设置units=3
# 的每个 LSTM unit 的计算输出结果。
print(output_data_all)
