# -*- coding:utf-8 -*-
"""
@author:Mr.Five
@email:wuwenfu5@qq.com
@num&WeChat:+86-15241192213
@file:testv.py
@time:18-4-7下午8:33
"""

import numpy as np
import cv2
import time

# cap = cv2.VideoCapture(r'/home/wuwenfu5/PycharmProjects/Three_people_cross.mp4')
cap = cv2.VideoCapture(r'/media/wuwenfu5/Win&Ubuntu/Python_/Material/Fast_wending.mp4')
# cap = cv2.VideoCapture(r'/media/wuwenfu5/Win&Ubuntu/Python_/Material/DJI_4.mp4')
# cap = cv2.VideoCapture(r'/media/wuwenfu5/Win&Ubuntu/Python_/Material/swimming03.mp4')

# cap = cv2.VideoCapture(0)


if cap.isOpened() is False:
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

time_kaishi = time.perf_counter()

while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        keycode = cv2.waitKey(1) & 0xff
        if keycode != 0xff:
            print(keycode, '\n', chr(keycode))
            break

        time.sleep(0.05)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fgmask = fgbg.apply(gray)
        cv2.imshow('Foreground', fgmask)
        cv2.moveWindow('Foreground', 740, 0)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel3x3)
        cv2.imshow('Opening', fgmask)
        cv2.moveWindow('Opening', 740, 537)

        ret, binary = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel6x6, iterations=2)

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)
        frame_RGB = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        frame_RGB = cv2.drawKeypoints(frame_RGB, keypoints, np.array([]), (0, 255, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for index in range(np.size(keypoints)):
            x = int(keypoints[index].pt[0])
            y = int(keypoints[index].pt[1])
            w = h = int(keypoints[index].size)
            cv2.rectangle(frame, (x - int(0.6 * w), y - int(1.2 * h)), (x + int(0.6 * w), y + int(1.2 * h)),
                          (0, 255, 0), 1)

        str_t = time.strftime('%Y-%m-%d %H:%M:%S')
        time_now = time.perf_counter()
        time_delta = time_now - time_last
        time_last = time_now
        fps = 1.0 / time_delta
        fps_t = str('fps: %.3f' % fps)
        det_t = str('calculate time:%.3f s' % time_delta)

        frame_count += 1  # 帧数
        fra_t = str('frame count:%d' % frame_count)

        cv2.putText(frame, fra_t, (450, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, fps_t, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, det_t, (0, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, str_t, (450, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        # ret, frame = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                               cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('Original', frame)
        cv2.moveWindow('Original', 100, 0)
        cv2.imshow('Closing And Blob', frame_RGB)
        cv2.moveWindow('Closing And Blob', 100, 537)

    else:
        print('get frame error!')
        break

print('process time:%0.3f' % (time.perf_counter() - time_kaishi))

cap.release()
cv2.destroyAllWindows()
