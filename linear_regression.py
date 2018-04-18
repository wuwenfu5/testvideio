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
