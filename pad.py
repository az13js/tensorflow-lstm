# -*- coding: UTF-8 -*-

import pandas as pd

print('VERSION')
print(pd.__version__)

x = pd.read_csv('test.csv')
print(x)
