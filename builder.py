# -*- coding: UTF-8 -*-

# 执行这个文件将会创建 data.csv

import pandas as pd
import numpy as np

def build(file, lines = 3, columns = 4):
    data = []
    line_data = []
    for i in range(columns):
        line_data.append(1000.0)
    data.append(line_data)
    for i in range(lines - 1):
        line_data = []
        for j in range(columns):
            line_data.append(float(data[i][j] + 0.5 * np.random.normal(0, 1, 1)))
        data.append(line_data)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(file)

if __name__ == "__main__":
    build('data.csv', 5000, 6)