# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:linear_regression.py
@time:18-4-18下午5:59
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 产生训练集
train_X = np.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_train_samples = train_X.shape[0]
print('训练样本数量:', n_train_samples)
# 测试样本
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
n_test_samples = test_X.shape[0]
print('测试样本数量:', n_test_samples)

# 显示原始数据分布
plt.scatter(train_X, train_Y, color='red', marker='o', label='Original Train Points')
plt.scatter(test_X, test_Y, color='blue', marker='*', label='Original Test Points')
plt.legend()
plt.show()

print('~~~~~~~~开始设计计算图~~~~~~~~')
with tf.Graph().as_default():
    # 输入占位符
    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, name='X')
        Y_true = tf.placeholder(tf.float32, name='Y_ture')

    # 模型参数变量
    with tf.name_scope('Inference'):
        W = tf.Variable(tf.zeros([1]), name='Weight')
        b = tf.Variable(tf.zeros([1]), name='Bias')

        # inference: y = wx + b
        Y_pred = tf.add(tf.multiply(X, W), b)

    # 添加损失
    with tf.name_scope('Loss'):
        TrainLoss = tf.reduce_mean(tf.pow((Y_true - Y_pred), 2)) / 2

    with tf.name_scope('Train'):
        # 创建一个梯度下降优化器
        Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        # 定义训练节点将梯度下降法应用于Loss
        TrainOp = Optimizer.minimize(TrainLoss)

    # 添加评估节点
    with tf.name_scope('Evaluate'):
        EvalLoss = tf.reduce_mean(tf.pow((Y_true - Y_pred), 2)) / 2

    # 保存计算图
    writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
    writer.close()
