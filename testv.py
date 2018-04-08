# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:testv.py
@time:18-4-7下午8:33
"""

# import numpy as np
import cv2
# import time

img = cv2.imread('/home/wuwenfu5/PycharmProjects/testcv1.jpg')

cv2.imshow('testcv',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture(r"/home/wuwenfu5/PycharmProjects/Three_people_cross.mp4")
# cap = cv2.VideoCapture(0)



if False == cap.isOpened():
    print('capture is not opened')
else:
    print('capture is opened')


kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel6x6 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200

params.minDistBetweenBlobs = 30

# 检测白色
params.filterByColor = True
params.blobColor = 255

# Filter by Area. 面积
params.filterByArea = True
params.minArea = 100
params.maxArea = 2500

# Filter by Circularity 圆度
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity 凸性
params.filterByConvexity = False
params.minConvexity = 0.3

# Filter by Inertia 惯性
params.filterByInertia = False
params.minInertiaRatio = 0.01

time_last = -1.0  # 避免刚运行程序时时间差为0，
frame_count = 0

while (1):
    ret, frame = cap.read()

    if ret == True:
        keycode = cv2.waitKey(1) & 0xff
        if keycode != 0xff:
            print(keycode, '\n', chr(keycode))
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fgmask = fgbg.apply(gray)
        cv2.imshow('Foreground', fgmask)

    else:
        print('get frame error!')
        break

cap.release()
cv2.destroyAllWindows()
