# -*- coding: UTF-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TrainDataBuilder:
    def __init__(self, file_name, input_length = 10, output_length = 1):
        self.file_name = file_name
        self.data_frame = pd.read_csv(file_name)
        self.input_length = input_length
        self.output_length = output_length

    def train_data(self):
        (length, columns) = self.data_frame.shape
        x = []
        y = []
        for start_line in range(len(self.data_frame) - self.input_length - self.output_length + 1):
            x.append(np.array(self.data_frame.iloc[start_line : start_line + self.input_length, 1 : columns]))
            y.append(np.array(self.data_frame.iloc[start_line + self.input_length : start_line + self.input_length + self.output_length, 1]))
        return (np.asarray(x), np.asarray(y))

if __name__ == "__main__":
    model = tf.keras.models.load_model("model.h5", compile = True)
    # 打印模型
    model.summary()

    # 构造训练数据
    tdb = TrainDataBuilder("normalization.csv", 10)
    (train_x, train_y) = tdb.train_data()

    # 训练模型
    history_data = model.fit(train_x, train_y, batch_size = 100, epochs = 3000, verbose = 0, shuffle = True)
    model.save("finally.h5", overwrite=True, include_optimizer=True)

    # 显示训练过程的误差变化
    plt.plot(range(len(history_data.history["loss"])), history_data.history["loss"])
    plt.show()
