#导出avi视频的每一帧，并保存在images路径下
# -*- encoding: utf-8 -*-
import cv2
import os
 
images = './image/'  ##保存路径
if not os.path.exists(images):
    os.mkdir(images)
 
cap = cv2.VideoCapture("save(1).avi") #视频位置
c=0
while(1):
    success, frame = cap.read()
    if success:
        img = cv2.imwrite(images+str(c) + '.jpg',frame)
        c=c+1
    else:
        break
cap.release()
