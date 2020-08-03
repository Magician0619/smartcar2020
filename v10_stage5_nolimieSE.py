# -*- coding:utf-8 -*-
'''
Filename: /Users/kingtous/PycharmProjects/neuq_car_upper/Auto_Driver_client_v2.py
Path: /Users/kingtous/PycharmProjects/neuq_car_upper
Created Date: Friday, July 17th 2020, 9:24:23 am
Author: Kingtous
Version: v5
Change: add video support
"""
get image from camera:/dev/video2  424*240
deal 128 *128     256*256
get the angle     object_detect
"""
Copyright (c) 2020 Kingtous' 2020
'''

from user import user_cmd
from datetime import datetime
from collections import namedtuple
from PIL import ImageDraw
from PIL import ImageFont
import os
import v4l2capture
from ctypes import *
import struct
import array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import getopt
import sys
import select
import termios
import tty
import threading
import paddlemobile as pm
from paddlelite import *
import codecs
#import paddle
import multiprocessing
#import paddle.fluid as fluid
#from IPython.display import display
import math
import functools
from PIL import Image
from PIL import ImageFile
import threading
from central import Central
ImageFile.LOAD_TRUNCATED_IMAGES = True

#script,vels,save_path= argv
# GLOBAL DEFINITION START
path = os.path.split(os.path.realpath(__file__))[0]+"/.."
opts, args = getopt.getopt(argv[1:], '-hH', ['save_path=', 'vels=', 'camera='])
# img savepath,camera
camera = "/dev/video2"
save_path = 'model_infer'
# car character
vels = 1560
limit_vels = 1520
crop_size = 128
recog_rate = 0.5
curent_speed = 1500
# recog classes
classes = 10
label_dict = {0: "n/limit", 1: "limit", 2: "park", 3: "red",
              4: "green", 5: "zebra", 6: "left", 7: "tank",
              8:"straight",9:"car"}
DEBUG_MODE = False


def output(s):
    global DEBUG_MODE
    if DEBUG_MODE == True:
        print(s)
# GLOBAL DEFINITION END

for opt_name, opt_value in opts:
    if opt_name in ('-h', '-H'):
        output("python3 Auto_Driver.py --save_path=%s  --vels=%d --camera=%s " %
               (save_path, vels, camera))
        exit()

    if opt_name in ('--save_path'):
        save_path = opt_value

    if opt_name in ('--vels'):
        vels = int(opt_value)

    if opt_name in ('--camera'):
        camera = opt_value


def load_image(cap):

    lower_hsv = np.array([156, 43, 46])
    upper_hsv = np.array([180, 255, 255])
    lower_hsv1 = np.array([0, 43, 46])
    upper_hsv1 = np.array([10, 255, 255])
    ref, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    mask1 = cv2.inRange(hsv, lowerb=lower_hsv1, upperb=upper_hsv1)
    mask = mask0 + mask1
    img = Image.fromarray(mask)
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.transpose((2, 0, 1))
    img = img[(2, 1, 0), :, :] / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

frame_count = 0

def dataset(video):
    global frame_count
    lower_hsv = np.array([25, 75, 190])
    upper_hsv = np.array([40, 255, 255])

    select.select((video,), (), ())

    image_data = video.read_and_queue()

    frame = cv2.imdecode(np.frombuffer(
        image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    '''load  128*128'''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    mask = mask0  # + mask1
    img = Image.fromarray(mask)
    img = img.resize((128, 128), Image.ANTIALIAS)
    #img = cv2.resize(img, (128, 128))
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0
    #output("Before func_expand image_shape:",img.shape)
    img = np.expand_dims(img, axis=0)
    #output("vedio image_shape:",img.shape)
    '''object   256*256'''
    img_256 = Image.fromarray(frame)
    frame_count += 1
    return img_256, img, frame

# *************
# car line
# *************
predictor_index = 0     #0是U 1是半U 2是限速 3是环 4是直角弯道
vels_index = (1560,1558,1525,1540,1555)
def load_angle_func():
    def p0(v):
        return 2000 * v + 500
    def p1(v):
        return 2000 * v + 500
    def p2(v):
        return 1800 * v + 600
    def p3(v):
        return 2400 * v + 300
    def p4(v):
        return 1600 * v + 700
    return (p0,p1,p2,p3,p4)


def load_model():

    path = "/home/root/workspace/deepcar/deeplearning_python/model/"
    valid_places = (
        Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
        Place(TargetType.kHost, PrecisionType.kFloat),
        Place(TargetType.kARM, PrecisionType.kFloat),
    )

    model_dir = "model_infer_deepvertical"
    #input("vertical\n\n\n")
    config4 = CxxConfig()
    config4.set_model_file(path +model_dir + "/model")
    config4.set_param_file(path +model_dir + "/params")
    #config.model_dir = model_dir
    config4.set_valid_places(valid_places)
    vertical_predictor = CreatePaddlePredictor(config4)

    #input("halfu\n\n\n")
    model_dir = "model_infer_1560deep"
    config = CxxConfig()
    config.set_model_file(path + model_dir + "/model")
    config.set_param_file(path + model_dir + "/params")
    #config.model_dir = model_dir
    config.set_valid_places(valid_places)
    halfu_predictor = CreatePaddlePredictor(config)
   
    config1 = CxxConfig()
    #input("U\n\n\n")
    model_dir = "model_infer_1560deep"
    config1.set_model_file(path + model_dir + "/model")
    config1.set_param_file(path + model_dir + "/params")
    #config.model_dir = model_dir
    config1.set_valid_places(valid_places)
    u_predictor = CreatePaddlePredictor(config1)

    #input("circle\n\n\n")
    model_dir = "model_infer_1540circledeep"
    config3 = CxxConfig()
    config3.set_model_file(path +model_dir + "/model")
    config3.set_param_file(path +model_dir + "/params")
    #config.model_dir = model_dir
    config3.set_valid_places(valid_places)
    circle_predictor = CreatePaddlePredictor(config3)

    model_dir = "model_infer_0729part_limit"
    #input("limit\n\n\n")
    config2 = CxxConfig()
    config2.set_model_file(path + model_dir + "/limit_model")
    config2.set_param_file(path +model_dir + "/limit_params")
    #config.model_dir = model_dir
    config2.set_valid_places(valid_places)

    limit_predictor = CreatePaddlePredictor(config2)
    return (u_predictor,halfu_predictor,limit_predictor,circle_predictor,vertical_predictor)


def predict(predictor, image, z):
    img = image

    i = predictor.get_input(0)
    i.resize((1, 3, 128, 128))
    #output("predict img.shape:",img.shape)
    #output("predict z.shape:",z.shape)
    z[0, 0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
    z = z.reshape(1, 3, 128, 128)
    frame1 = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    #cv2.imwrite("first_frame.jpg", frame1)
    i.set_data(z)

    predictor.run()
    out = predictor.get_output(0)
    score = out.data()[0][0]
    #output(out.data()[0])
    return score


# *************
# object detect
# *************

train_parameters = {
    "train_list": "train.txt",
    "eval_list": "eval.txt",
    "class_dim": -1,
    "label_dict": {},
    "num_dict": {},
    "image_count": -1,
    "continue_train": True,
    "pretrained": False,
    "pretrained_model_dir": "./pretrained-model",
    "save_model_dir": "./yolo-model",
    "model_prefix": "yolo-v3",
    "freeze_dir": "freeze_model",
    # "freeze_dir": "../model/tiny-yolov3",
    "use_tiny": True,
    "max_box_num": 20,
    "num_epochs": 80,
    "train_batch_size": 32,
    "use_gpu": False,
    "yolo_cfg": {
        "input_size": [3, 448, 448],
        "anchors": [7, 10, 12, 22, 24, 17, 22, 45, 46, 33, 43, 88, 85, 66, 115, 146, 275, 240],
        "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    },

    "yolo_tiny_cfg": {
        "input_size": [3, 256, 256],
        "anchors": [6, 8, 13, 15, 22, 34, 48, 50, 81, 100, 205, 191],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,
    "nms_top_k": 300,
    "nms_pos_k": 300,
    "valid_thresh": 0.01,
    "nms_thresh": 0.45,

    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },

    "sgd_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [30, 50, 65],
        "lr_decay": [1, 0.5, 0.25, 0.1]
    },

    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "min_loss": 2.5,
        "min_curr_map": 0.84
    }
}


def init_train_parameters():

    # os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    file_list = "./data/data6045/train.txt"
    # os.path.join(train_parameters['data_dir'], "label_list")
    label_list = "./data/data6045/label_list"
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['num_dict'][index] = line.strip()
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)


def read_image(buffer):
    #lock.acquire()
    # origin: ndarray
    origin = Image.open(buffer)
    #img = resize_img(origin, target_size)
    #decode_img = cv2.imdecode(np.frombuffer(buffer,np.uint8),-1)
    #img = cv2.resize(decode_img,(256,256))
    img = origin.resize((256,256), Image.BILINEAR)   #######resize 256*256
    # print(type(img))
    #lock.release()

    #origin = image
    #img = resize_img(origin, target_size)

    # img = origin.resize((256,256), Image.BILINEAR)   #######resize 256*256

    # added
    #img = cv2.resize(origin, (256,256))

    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    #img = np.array(img).astype('float32')
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return img

def load_model_detect():
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    target_size = yolo_config['input_size']
    anchors = yolo_config['anchors']
    anchor_mask = yolo_config['anchor_mask']
    label_dict = train_parameters['num_dict']
    #output("label_dict:", label_dict)
    class_dim = train_parameters['class_dim']
    # output("class_dim:",class_dim)

    path1 = train_parameters['freeze_dir']
    model_dir = path1
    pm_config1 = pm.PaddleMobileConfig()
    pm_config1.precision = pm.PaddleMobileConfig.Precision.FP32  # ok
    pm_config1.device = pm.PaddleMobileConfig.Device.kFPGA  # ok
    #pm_config.prog_file = model_dir + '/model'
    #pm_config.param_file = model_dir + '/params'
    pm_config1.model_dir = model_dir
    pm_config1.thread_num = 4
    predictor1 = pm.CreatePaddlePredictor(pm_config1)
    # Cxx
    # valid_places =   (
    # 	Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
    # 	Place(TargetType.kFPGA, PrecisionType.kInt8),
    # 	Place(TargetType.kFPGA, PrecisionType.kInt16),
    # );
    # model_dir = "/home/root/workspace/deepcar/deeplearning_python/model/detect_model_infer"
    # config = CxxConfig();
    # config.set_model_file(model_dir+"/model");
    # config.set_param_file(model_dir+"/params");
    # #config.model_dir = model_dir
    # config.set_valid_places(valid_places);
    # predictor = CreatePaddlePredictor(config);

    return predictor1

# process frame adding labels,scores,boxes
# return frame

def process_frame(img, labels, scores, boxes, need_raw=True):
    # need_raw -> True, output raw video, no prediction text and rectangle
    x_rate = 320.0 / 608
    y_rate = 240.0 / 608
    #boxes = boxes[:,2:].astype('float32')
    d = img
    if not need_raw:
        output(boxes)
        for label, box, score in zip(labels, boxes, scores):
            # output("label:",label_dict[int(label)])
            if score < recog_rate:
                continue
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            d = cv2.rectangle(img, (int(box[0]*x_rate), int(box[1]*y_rate)),
                            (int(box[2]*x_rate), int(box[3]*y_rate)), (255, 0, 0), 1)
            d = cv2.putText(img, label_dict[int(label)]+":"+str(score), (int(
                box[0]*x_rate), int(box[3]*y_rate)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
    return d


# queue thread
def queueThread(lib,pipe):
    global curent_speed
    while 1:
        p1,p2,t = pipe.recv()
        time.sleep(t)
        lib.send_cmd(p1,p2)
        if (p1 == 0):
            curent_speed = p2
        print("sent ",p1,p2,"delay time:",t,"s")

if __name__ == "__main__":
    cout = 0
    save_path = path + "/model/" + save_path
    # 视频
    video = v4l2capture.Video_device(camera)
    video.set_format(320, 240, fourcc='MJPG')
    video.create_buffers(1)
    video.queue_all_buffers()
    video.start()
    #
    predictor_list = load_model()
    angle_list = load_angle_func()
    '''##########################################################object  detect##########################################################'''
    init_train_parameters()
    detect_predictor = load_model_detect()

    vel = int(vels)
    lib_path = path + "/lib" + "/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    
    car = "/dev/ttyUSB0"
    z = np.zeros((1, 128, 128, 3))

    if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
        raise
        pass
    central = Central(lib,vels,vels_list=vels_index,init_predictor_index = 0)
    # most possible status
    # VIDEO
    out = cv2.VideoWriter(
        'save.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))

    # cv2.VideoWriter
    # labels in no_limit(0),limit(1),park(2),red(3),green(4),zebra(5),block(6)
    # DEFINITION END
    NO_LIMIT = 0
    LIMIT = 1
    PARK = 2
    RED = 3
    GREEN = 4
    ZEBRA = 5
    LEFT = 6
    BARRIER = 7
    #
    STRAIGHT = 8
    CAR = 9
    # 急刹车
    STOP = 10
    # DEFINITION
    RED_DETECTED = False
    try:
        while 1:
            count = 0
            yuzhi_0 = 0
            yuzhi_3 = 0
            yuzhi_4 = 0
            # SIGN DEFINITION START
            # judge zebra line
            zebra_line_detected = False
            # reset ZEBRA and RED and GREEN
            STATUS_LIMIT = False
            STATUS_PARK = False
            LAST_FINAL_LABEL = NO_LIMIT

            NO_LIMIT_ALREADY_DETECTED = False
            NO_LIMIT_ALREADY_LAST = time.time()
           
            z = np.zeros((1, 128, 128, 3))
            while 1:
                t1 = time.time()
                ZEBRA_SIGN = False
                RED_SIGN = False
                GREEN_SIGN = False
                LIMIT_SIGN = False
                NO_LIMIT_SIGN = False
                BARRIER_SIGN = False
                PARK_SIGN = False
                # true if NO LIMIT SIGN detected
                nowtime = time.time()
                origin, img, frame = dataset(video)
                # convert bgr to rgb manually
                color_b, color_g, color_r = origin.split()
                origin = Image.merge("RGB", (color_r, color_g, color_b))

                # STEP 1: ANGLE
                angle = predict(predictor_list[predictor_index], img, z)
                try:
                    true_angle = int(angle_list[predictor_index](angle))
                except:
                    true_angle = 1500
                # TEST carline only
                #lib.send_cmd(true_angle,0)
                #continue
                if LAST_FINAL_LABEL == STOP:
                    lib.send_cmd(true_angle,PARK)
                else:
                    lib.send_cmd(true_angle,int(LAST_FINAL_LABEL))
                
                # DISABLE DETECTION
                # output("angle:",angle)
                # lib.send_cmd(int(700+1600*angle),0)
                # continue
                # tensor_img,img= read_image()  #  resize image
                tensor_img = origin.resize((256,256), Image.BILINEAR)   #######resize 256*256
                # tensor_img.mode = 'BGR'
                # if tensor_img.mode != 'RGB':
                #     #    tensor_img = cv2.cvtColor(Image.fromarray(tensor_img),cv2.COLOR_RGB2BGR)
                #     tensor_img = tensor_img.convert('RGB')
                #     #    print(type(tensor_img))
                # print("mode:",tensor_img.mode)
                tensor_img = np.array(tensor_img).astype('float32')#.transpose((2, 0, 1))  # HWC to CHW
                tensor_img -= 127.5
                tensor_img *= 0.007843
                show_img = tensor_img
                tensor_img = tensor_img[np.newaxis, :]

                # print("shape", tensor_img.shape)
                # print("origin",tensor_img)
                # print("after",tensor_img)
                tensor = pm.PaddleTensor()
                tensor.dtype = pm.PaddleDType.FLOAT32
                tensor.shape = (1, 3, 256, 256)
                tensor.data = pm.PaddleBuf(tensor_img)
                paddle_data_feeds1 = [tensor]
                count += 1
                outputs1 = detect_predictor.Run(paddle_data_feeds1)
                #output("outputs1 value:", str(outputs1))

                # assert len(
                #     outputs1) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'
                bboxes = np.array(outputs1[0], copy=False)
                # output("bboxes.shape",bboxes.shape)

                t_labels = []
                t_scores = []
                t_boxes = []
                center_x = []
                center_y = []

                # starting judge labels
                final_label = NO_LIMIT
                if STATUS_LIMIT == True:
                    final_label = LIMIT
                final_score = 0.0

                if len(bboxes.shape) == 1:
                    output("No object found in video")
                    STATE_value = False
                else:
                    STATE_value = False
                    labels = bboxes[:, 0].astype('int32')
                    # scores -> angle , true_angle = 500 + 2000*scores
                    scores = bboxes[:, 1].astype('float32')
                    # pixels position
                    boxes = bboxes[:, 2:].astype('float32')
                    # print("labels:",str(labels))
                    # print("scores:",str(scores))
                    # print("boxes:", str(boxes))
                    frame = process_frame(frame, labels, scores, boxes, need_raw=False)
                    # start algorithm
                    # <- final_label
                    for i in range(len(labels)):
                        if scores[i] >= recog_rate:
                            # t_labels.append(label_dict[labels[i]])
                            # read sign
                            if labels[i] == ZEBRA:
                                ZEBRA_SIGN = True
                            elif labels[i] == RED and RED_DETECTED == False:
                                RED_SIGN = True
                                RED_DETECTED = True
                            elif labels[i] == GREEN:
                                GREEN_SIGN = True
                            elif labels[i] == LIMIT:
                                LIMIT_SIGN = True
                            elif labels[i] == NO_LIMIT:
                                NO_LIMIT_SIGN = True
                                NO_LIMIT_ALREADY_DETECTED = True
                            elif labels[i] == BARRIER:
                                BARRIER_SIGN = True
                            elif labels[i] == PARK:
                                PARK_SIGN = True
                            else:
                                # process
                                if scores[i] > final_score:
                                    final_label = labels[i]
                                    final_score = scores[i]
                                # collect infos, of no use currently
                                t_labels.append(labels[i])
                                # output("t_labels:",str(t_labels))
                                t_scores.append(scores[i])
                                # output("t_scores:",str(t_scores))
                                center_x.append(
                                    int((boxes[i][0]+boxes[i][2])/2))
                                center_y.append((boxes[i][1]+boxes[i][3])/2)
                                #output("the center coordinate of object:", center_x, "   ", center_y)

                    STATE_value = True

                ################################################################################################
                # preprocessing data (data、sign)
                # TODO BLOCK Judgement
                # TODO
                # calculate true angle
                # TODO 角度突变

                # Judge SIGN
                delay_time = 0
                if RED_SIGN == True:  # and ZEBRA_SIGN == True:
                    # cv2.imwrite("red.jpg",tensor_img)
                    # V2: only recognize both SIGNs
                    final_label = STOP
                    STATUS_PARK = True
                if PARK_SIGN == True:
                    # P
                    final_label = PARK
                    STATUS_PARK = True
                    delay_time = 0.4
                if GREEN_SIGN == True and STATUS_PARK == True:
                    # cv2.imwrite("green.jpg",tensor_img)
                    if STATUS_LIMIT:
                        final_label = LIMIT
                    else:
                        final_label = NO_LIMIT
                    STATUS_PARK = False
                    if predictor_index == 0:
                        predictor_index = 1
                    
                if LIMIT_SIGN == True:
                    final_label = LIMIT
                    STATUS_PARK = False


                if NO_LIMIT_SIGN == True and STATUS_LIMIT == True:
                    final_label = NO_LIMIT
                    # prevent error
                    STATUS_PARK = False

                if final_label == LEFT:
                    if predictor_index == 3:
                        #lib.send_cmd(0,1560)
                        predictor_index = 4
                        delay_time = 0.5

                # start sending proccessed command
                #output("sending: angle: %d, label: %d" % (true_angle, final_label))
                # user_cmd(STATE_value,t_labels,t_scores,center_x,center_y,vel,a)
                if STATUS_PARK == True and PARK_SIGN == False and RED_SIGN == False:
                    final_label = PARK

                # if STATUS_LIMIT == True :
                #     final_label = LIMIT
                #     lib.send_cmd(0,limit_vels)
                #     print("sent speed:",limit_vels)

                if BARRIER_SIGN == True and STATUS_PARK != False:
                    final_label = BARRIER

                # final sent process
                if final_label == LIMIT and LIMIT_SIGN == True and STATUS_LIMIT == False:
                    STATUS_LIMIT = True
                    # delay_p1.send((0,limit_vels,0))
                    # delay_p1.send((true_angle,final_label,0))
                    curent_speed = limit_vels
                    delay_time = 0.5
                    if predictor_index == 1:
                        predictor_index = 2
                #if time.time() - NO_LIMIT_ALREADY_LAST  

                if NO_LIMIT_ALREADY_DETECTED == True and NO_LIMIT_SIGN == False and STATUS_LIMIT == True:
                    STATUS_LIMIT = False
                    # delay_p1.send((0,vels,0))
                    # delay_p1.send((true_angle,final_label,0))
                    curent_speed = vels
                    delay_time = 2
                    if predictor_index == 2:
                        predictor_index = 3
                    NO_LIMIT_ALREADY_DETECTED = False

                if STATUS_PARK == True and final_label == LIMIT:
                    final_label = PARK

                if final_label == NO_LIMIT and STATUS_LIMIT == True:
                    final_label = LIMIT
                    delay_time = 1.5
                
                if final_label == STRAIGHT:
                    if STATUS_LIMIT == True:
                        final_label = LIMIT
                    else:
                        final_label = NO_LIMIT

                if final_label == STOP:
                	STATUS_PARK = True


                if LAST_FINAL_LABEL == STOP and final_label == STOP:
                    final_label = PARK

	
                central.request(true_angle,int(final_label),delay_time)
                if STATUS_PARK == False:
                    print(vels_index[predictor_index])
                    central.change_speed(predictor_index,delay_time)
                #central.change_speed(predictor_index)

                if final_label == PARK and LAST_FINAL_LABEL == STOP:
                    LAST_FINAL_LABEL = STOP
                else:
                    LAST_FINAL_LABEL = final_label

                #if delay_time > 0:
                #    input("delay_time:"+str(delay_time))
                #print("is parking?",STATUS_PARK)
                #print("is limit?",STATUS_LIMIT)
                #print("label:",final_label)
                #print("delay:",delay_time)
                
                t2 = time.time()
                cout = cout + 1
                #if delay_time > 0:
                #    input("delayed")
                #print("convert time:", t2-t1)
                # output VIDEO
                cv2.putText(frame,"line pt: " + "lp" if STATUS_LIMIT else "p",(0,180),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                cv2.putText(frame,str(true_angle) + ":" + str(final_label),(0,200),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                cv2.putText(frame,"is parking?:" + str(STATUS_PARK),(0,220),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                cv2.putText(frame,"index:" + str(predictor_index),(0,160),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                cv2.putText(frame,"s:" + str(vels_index[predictor_index]),(0,140),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                cv2.putText(frame,"frame:" + str(frame_count),(0,20),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                cv2.putText(frame,"delay:" + str(delay_time),(0,40),cv2.FONT_HERSHEY_PLAIN,1.5, (0, 255, 0), 1)
                # VIDEO Write
                out.write(frame)
                # if final_label == 2:
                #    input("stop")
                # if final_label == 1:
                #    input("limit")
                # output(cout)
                
                #output("the time of predict:",time.time()-nowtime)
                
    except KeyboardInterrupt as e:
        output("keyboard detected")
        #VIDEO
        out.release()
        # stop car
        lib.send_cmd(1500, 2)
        exit(0)
    finally:
        output('exit')
