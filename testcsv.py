# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:testcsv.py
@time:18-4-13下午1:55
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot
from numpy.linalg import inv
from numpy import mat


df = pd.read_csv(r'/home/wuwenfu5/PycharmProjects/Advertising.csv', header=None)
# print(df.head(10))

X = df.loc[1:10, 1:3].values
Y = df.loc[1:10, 4].values

for index in range(len(Y)):
    X[index, 0] = float(X[index, 0])
    X[index, 1] = float(X[index, 1])
    X[index, 2] = float(X[index, 2])
    Y[index] = float(Y[index])

Z = X.T.dot(X)
print(Z)
# Z = Z.reshape(3, 3)
print(inv(Z))
# Z = [[1, 2, 3], [3, 4, 5], [6, 7, 8]]
# print(Z)
# print(inv(Z))

# theta = dot(dot(inv(dot(X.T, X)), X.T), Y)
# print(theta)

plt.xlabel('XXX')
plt.ylabel(df.loc[0, 4])


plt.scatter(X[:, 0], Y, c='red', marker='.', label=df.loc[0, 1],s=5)
plt.scatter(X[:, 1], Y, c='green', marker='*', label=df.loc[0, 2],s=5)
plt.scatter(X[:, 2], Y, c='blue', marker='v', label=df.loc[0, 3],s=5)


plt.legend(loc='upper left')


# plt.show()
