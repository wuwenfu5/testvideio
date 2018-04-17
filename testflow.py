# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:testflow.py
@time:18-4-17下午3:59
"""

import tensorflow as tf


hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
