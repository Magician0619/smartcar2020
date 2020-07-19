#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import v4l2capture
import select
from ctypes import *
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import multiprocessing
import time
import getopt

path = os.path.split(os.path.realpath(__file__))[0]

save_name="img"

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("----- new folder -----")
    else:
        print('----- there is this folder -----')



def save_image_process(Camera):
    mkdir(path+"/data")
    mkdir(path+"/data/"+save_name)
    video = v4l2capture.Video_device(Camera)
    video.set_format(424,240, fourcc='MJPG')
    #video.set_format(160,120, fourcc='MJPG')
    video.create_buffers(1)
    video.queue_all_buffers()
    video.start() 
    imgInd = 0
    while  1:
        select.select((video,), (), ())        
        image_data = video.read_and_queue()
        frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        #cv2.imshow('video',frame)
        #cv2.imwrite(path+"/data/"+save_name+"/{}.jpg".format(imgInd), frame)
        a.value = imgInd
        print("imgInd=",imgInd)
        imgInd+=1
        time.sleep(0.5)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break 


def save_vedio_process(Camera):
    mkdir(path+"/data")
    video = v4l2capture.Video_device(Camera)

    video.create_buffers(1)
    video.queue_all_buffers()
    video.start() 
    
    video.set_format(424,240, fourcc='XVID')
    #video.set_format(160,120, fourcc='MJPG')
    out = cv2.VideoWriter('save.avi',vedio,20,(420,240))   #name,frame rate,format

    while(True):
        select.select((video,), (), ())
        image_data = video.read_and_queue()
        frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        out.write(frame)
        
        if key & 0xFF == ord('q'):
            break 
        

    out.release()



 
 


if __name__ == '__main__':
    save_vedio_process("/dev/video2")
   

