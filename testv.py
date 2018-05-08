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
import os

# cap = cv2.VideoCapture(r'/media/wuwenfu5/Win_Ubuntu_Swap/Python_/Material/MyMOT/shitang10.AVI')
cap = cv2.VideoCapture(r'/media/wuwenfu5/Win_Ubuntu_Swap/Python_/Material/Many.mp4')
body_cascade = cv2.CascadeClassifier(
    r'/usr/local/share/OpenCV/haarcascades/haarcascade_fullbody.xml')
# cap = cv2.VideoCapture(0)


if cap.isOpened() is False:
    print('capture is not opened')
else:
    print('capture is opened')

kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel6x6 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=True)
# fgbg = cv2.createBackgroundSubtractorKNN(history=200, detectShadows=True)

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
params.minArea = 150
params.maxArea = 50000

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
frame_count = 1

time_kaishi = time.perf_counter()

fps_ = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frames_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frames_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('fps=', fps_)
print('frames= ', frames)
print('frames_wxh=', frames_w, 'x', frames_w)

# tracker = cv2.MultiTracker_create()

results = []
detections_out = []
while cap.isOpened():
    ret, frame = cap.read()

    if ret == True:
        keycode = cv2.waitKey(1) & 0xff
        if keycode != 0xff:
            print(keycode, '\n', chr(keycode))
            break

        # time.sleep(0.05)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fgmask = fgbg.apply(gray)
        cv2.imshow('Foreground', fgmask)
        cv2.moveWindow('Foreground', 740, 0)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel3x3)
        cv2.imshow('Opening', fgmask)
        cv2.moveWindow('Opening', 740, 537)

        ret, binary = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel6x6, iterations=2)

        # Copy the thresholded image.
        im_floodfill = binary.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = binary.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        binary = binary | im_floodfill_inv


        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)
        frame_RGB = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        frame_RGB = cv2.drawKeypoints(frame_RGB, keypoints, np.array([]), (0, 255, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        roi_count = 0

        for index in range(np.size(keypoints)):
            x = int(keypoints[index].pt[0])
            y = int(keypoints[index].pt[1])
            w = h = int(keypoints[index].size)

            # if frame_count % 10 == 0:
            roi_x1 = x - int(0.9 * w)
            roi_x2 = x + int(0.9 * w)
            roi_y1 = y - int(1.6 * h)
            roi_y2 = y + int(1.6 * h)
            if roi_x1 > 0 and roi_x2 > 0 and roi_y1 > 0 and roi_y2 > 0:
                roi = frame[roi_y1: roi_y2, roi_x1: roi_x2]
                # cv2.imshow('ROI', roi)
                # cv2.imwrite(str('./logs/img_%d_%d.png' % (frame_count, roi_count)), roi)
            roi_count += 1

            results.append([frame_count, index, roi_x1, roi_y1, int(1.8 * w), int(3.2 * h)])
            detections_out += [np.r_[(frame_count, index, roi_x1, roi_y1, int(1.8 * w), int(3.2 * h), 1, -1, -1, -1)]]
            cv2.rectangle(frame, (x - int(0.6 * w), y - int(1.2 * h)), (x + int(0.6 * w), y + int(1.2 * h)),
                          (0, 255, 0), 1)

        # _,cnts,_ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # for c in cnts:
        #     # if the contour is too small, ignore it
        #     if cv2.contourArea(c) < 300:
        #         continue
        #
        #     # 计算轮廓的边界框，在当前帧中画出该框
        #     x, y, w, h = cv2.boundingRect(c)
        #     print(cv2.boundingRect(c))
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

        # bodys = body_cascade.detectMultiScale(gray, 1.3, 3)
        # print(bodys)
        # for (x, y, w, h) in bodys:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Original', frame)
        cv2.moveWindow('Original', 100, 0)
        cv2.imshow('Closing And Blob', frame_RGB)
        cv2.moveWindow('Closing And Blob', 100, 537)

    else:
        print('get frame error!')
        break

print('process time:%0.3f' % (time.perf_counter() - time_kaishi))
# print(results)
# Store results.
f = open('./logs/det.txt', 'w')
for row in results:
    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
        row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

# output_filename = os.path.join('./logs', "det.npy")
# np.save(output_filename, np.asarray(detections_out), allow_pickle=False)
# print(detections_out)
cap.release()
cv2.destroyAllWindows()
