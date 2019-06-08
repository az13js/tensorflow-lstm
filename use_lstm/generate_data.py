# -*- coding: UTF-8 -*-

# 构造假数据

import pandas as pd
import numpy as np

def build(file, lines = 3, columns = 1):
    data = []
    line_data = []
    for i in range(columns):
        line_data.append(100.0)
    data.append(line_data)
    for i in range(lines - 1):
        line_data = []
        for j in range(columns):
            if 0 == j:
                line_data.append(float(data[i][j] + 0.1 * np.random.normal(0, 2)))
            else:
                line_data.append(float(data[i][0] + 0.1 * np.random.normal(0, 2)))
        data.append(line_data)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(file)

if __name__ == "__main__":
    build('data.csv', 100, 1)
    print("OK.")
