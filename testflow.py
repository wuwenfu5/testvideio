# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:testflow.py
@time:18-4-17下午3:59
"""

import tensorflow as tf
import numpy as np

# import os
#
# os.environ['TP_CPP_MIN_LOG_LEVEL'] = '3'
'''
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], 1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
'''
# print(tf.__version__)

a = tf.constant(5)
b = tf.constant(6)
with tf.Session() as sess:
    print('a=5, b=6')
    print('常量节点相加：%i' % sess.run(a + b))
    print('常量节点相乘：%i' % sess.run(a * b))

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    print('变量相加：%i' % sess.run(add, feed_dict={a: 5, b: 6}))
    print('变量相乘：%i' % sess.run(mul, feed_dict={a: 5, b: 6}))

matrix_a = tf.constant([[5, 6]])
matrix_b = tf.constant([[5], [6]])
product = tf.matmul(matrix_a, matrix_b)
with tf.Session() as sess:
    result = sess.run(product)
    print('矩阵乘法：', result)

writer = tf.summary.FileWriter(logdir='logs', graph=tf.get_default_graph())
writer.flush()
