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

X = df.loc[1:100, 1:3].values
Y = df.loc[1:100, 4].values

for index in range(len(Y)):
    X[index, 0] = float(X[index, 0])
    X[index, 1] = float(X[index, 1])
    X[index, 2] = float(X[index, 2])
    Y[index] = float(Y[index])

Z = mat(X.T.dot(X))
Z = [[2862763.310000001, 377273.22000000026, 470038.0499999996],
     [377273.22000000026, 82339.34999999998, 96599.56999999996],
     [470038.0499999996, 96599.56999999996, 155241.20000000004]]

# print(Z)
# Z = Z.I
Z = inv(Z)
# print(Z)


# theta = dot(dot(inv(dot(X.T, X)), X.T), Y)
theta = dot(dot(Z, X.T), Y)
print(theta)

theta = np.array([1., 1., 1.]).reshape(-1, 1)
# print(theta)
alpha = 0.00001
X0 = X[:, 0].reshape(-1, 1)
X1 = X[:, 1].reshape(-1, 1)
X2 = X[:, 2].reshape(-1, 1)
Y = Y.reshape(-1, 1)
temp = theta
for i in range(10000):
    temp[0] = theta[0] + alpha * np.sum((Y - dot(X, theta)) * X0) / 100.
    temp[1] = theta[1] + alpha * np.sum((Y - dot(X, theta)) * X1) / 100.
    temp[2] = theta[2] + alpha * np.sum((Y - dot(X, theta)) * X2) / 100.
    theta = temp
    # print(theta)

print(theta)
# print(X0)

# plt.xlabel('XXX')
# plt.ylabel(df.loc[0, 4])
#
# plt.scatter(X[:, 0], Y, c='red', marker='.', label=df.loc[0, 1], s=5)
# plt.scatter(X[:, 1], Y, c='green', marker='*', label=df.loc[0, 2], s=5)
# plt.scatter(X[:, 2], Y, c='blue', marker='v', label=df.loc[0, 3], s=5)
#
# x = np.arange(300)
# y = 5 + x * theta[0]
# # print(y)
#
# plt.scatter(x, y, c='black', marker='o', label=df.loc[0, 3], s=5)
#
# plt.legend(loc='upper left')
#
# plt.show()
