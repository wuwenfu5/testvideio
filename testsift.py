# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:testsift.py
@time:18-4-17上午11:59
"""

import cv2
import numpy as np


img = cv2.imread('/media/wuwenfu5/Win&Ubuntu/Python_/Material/wuwenfu80K.jpg')
new_w = 600
new_h = int(new_w * img.shape[0] / img.shape[1])
img = cv2.resize(img, (new_w, new_h))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)

sift = cv2.xfeatures2d.SIFT_create()
key_point = sift.detect(gray, None)
cv2.drawKeypoints(img, key_point, img)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

