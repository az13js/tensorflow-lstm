# -*- coding: UTF-8 -*-

import numpy as np
import math

x = np.array([1,2,3,4]) # int
y = 3 * x
print(y)
x[1] = 12.2
print(x[1])

print('####')

x = np.array([1,2,3,4], dtype=np.float64) # float
y = 3 * x
print(y)
x[1] = 12.2
print(x[1])