# -*- coding: UTF-8 -*-
import TextProcess
import Dataset
import Model
import numpy as np

# LSTM 处理的序列长度，根据多长的输入预测下一个值
sequeceLength = 2
# LSTM 层数
LSTMUnitNumber = 10

# 读取 test.txt 的内容并将其转换成字母向量列表的序列
context = TextProcess.getListSequenceFromFile("test.txt")
# 按照一定的长度作为输入序列，将 context 拆分成多个输入输出对，构成训练数据集
(inputs, outputs) = Dataset.sequenceFormat(context, sequeceLength, TextProcess.getVectorLength())
# 根据指定的参数，构造 LSTM 模型
model = Model.createLSTMModelForClassification(sequeceLength, TextProcess.getVectorLength(), LSTMUnitNumber)
# 配置训练模型所需的算法类型、参数和损失函数
model.compile(optimizer = Model.getOptimizers().Adam(0.0001), loss = "MSE")
#model.compile(optimizer = Model.getOptimizers().Adam(0.0001), loss = Model.softmaxCrossEntropy())
#model.compile(optimizer = Model.getOptimizers().Adam(0.0001), loss = Model.sigmoidCrossEntropy())
#model.compile(optimizer = Model.getOptimizers().Adam(0.0001), loss = Model.sparseSoftmaxCrossEntropy())

for i in range(100):
    # 训练模型
    model.fit(np.array(inputs), np.array(outputs), 1, 1);
    # 预测
    predicts = Dataset.predictSequence(model, context, sequeceLength, 10, TextProcess.getVectorLength())
    # 预测结果转换为字符串
    predictString = ""
    for wordVector in predicts:
        print(wordVector)
        predictString += TextProcess.listToChar(wordVector)
    # 输出预测结果
    print(predictString)
