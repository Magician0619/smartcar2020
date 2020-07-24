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

import os
import v4l2capture
from ctypes import *
import struct, array
from fcntl import ioctl
import cv2
import numpy as np
import time
from sys import argv
import getopt
import sys, select, termios, tty
import threading
import paddlemobile as pm
from paddlelite import *
import codecs
# import paddle
import multiprocessing
# import paddle.fluid as fluid
# from IPython.display import display
import math
import functools
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageFont
from PIL import ImageDraw
from collections import namedtuple
from datetime import datetime
from user import user_cmd

# script,vels,save_path= argv
### GLOBAL DEFINITION START
path = os.path.split(os.path.realpath(__file__))[0] + "/.."
opts, args = getopt.getopt(argv[1:], '-hH', ['save_path=', 'vels=', 'camera='])
# img savepath,camera
camera = "/dev/video2"
save_path = 'model_infer'
# car character
vels = 1550
crop_size = 128
recog_rate = 0.6
# recog classes
classes = 6
label_dict = {0:"no limit",1:"limit",2:"park",3:"red",4:"green",5:"zebra"}
### GLOBAL DEFINITION END

for opt_name, opt_value in opts:
    if opt_name in ('-h', '-H'):
        print("python3 Auto_Driver.py --save_path=%s  --vels=%d --camera=%s " % (save_path, vels, camera))
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


def dataset(video):
    lower_hsv = np.array([25, 75, 190])
    upper_hsv = np.array([40, 255, 255])

    select.select((video,), (), ())

    image_data = video.read_and_queue()

    frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    '''load  128*128'''

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
    mask = mask0  # + mask1
    img = Image.fromarray(mask)
    img = img.resize((128, 128), Image.ANTIALIAS)
    # img = cv2.resize(img, (128, 128))
    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img / 255.0;
    img = np.expand_dims(img, axis=0)
    print("vedio image_shape:", img.shape)
    '''object   256*256'''
    img_256 = Image.fromarray(frame)
    return img_256, img, frame;


# *************
# car line
# *************
def load_model():
    valid_places = (
        Place(TargetType.kFPGA, PrecisionType.kFP16, DataLayoutType.kNHWC),
        Place(TargetType.kHost, PrecisionType.kFloat),
        Place(TargetType.kARM, PrecisionType.kFloat),
    );
    config = CxxConfig();
    model = save_path;
    model_dir = model;
    config.set_model_file(model_dir + "/model");
    config.set_param_file(model_dir + "/params");
    # config.model_dir = model_dir
    config.set_valid_places(valid_places);
    predictor = CreatePaddlePredictor(config);
    return predictor;


def predict(predictor, image, z):
    img = image;

    i = predictor.get_input(0);
    i.resize((1, 3, 128, 128));
    print("predict img.shape:", img.shape)
    print("predict z.shape:", z.shape)
    z[0, 0:img.shape[1], 0:img.shape[2] + 0, 0:img.shape[3]] = img
    z = z.reshape(1, 3, 128, 128);
    frame1 = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("first_frame.jpg", frame1)
    i.set_data(z)

    predictor.run();
    out = predictor.get_output(0);
    score = out.data()[0][0];
    print(out.data()[0])
    return score;


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
    file_list = "./data/data6045/train.txt"  # os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    label_list = "./data/data6045/label_list"  # os.path.join(train_parameters['data_dir'], "label_list")
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


def load_model_detect():
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    target_size = yolo_config['input_size']
    anchors = yolo_config['anchors']
    anchor_mask = yolo_config['anchor_mask']
    label_dict = train_parameters['num_dict']
    # print("label_dict:", label_dict)
    class_dim = train_parameters['class_dim']
    # print("class_dim:",class_dim)

    path1 = train_parameters['freeze_dir']
    model_dir = path1
    pm_config1 = pm.PaddleMobileConfig()
    pm_config1.precision = pm.PaddleMobileConfig.Precision.FP32  ######ok
    pm_config1.device = pm.PaddleMobileConfig.Device.kFPGA  ######ok
    # pm_config.prog_file = model_dir + '/model'
    # pm_config.param_file = model_dir + '/params'
    pm_config1.model_dir = model_dir
    pm_config1.thread_num = 4
    predictor1 = pm.CreatePaddlePredictor(pm_config1)

    return predictor1


### process frame adding labels,scores,boxes
# return frame
def process_frame(img, labels, scores, boxes):
    x_rate = 320.0 / 608
    y_rate = 240.0 / 608
    # boxes = boxes[:,2:].astype('float32')
    d = img
    print(boxes)
    for label, box, score in zip(labels, boxes, scores):
        # print("label:",label_dict[int(label)])
        if score < recog_rate:
            continue
        d = cv2.rectangle(d, (int(box[0] * x_rate), int(box[1] * y_rate)),
                          (int(box[2] * x_rate), int(box[3] * y_rate)), (255, 255, 0), 1)
        d = cv2.putText(d, str(label) + ":" + str(score), (int(box[0] * x_rate), int(box[1] * y_rate)),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
    return d


### 线程区
def carLineProcess(predictor,pipe):
    z = np.zeros((1, 128, 128, 3))
    while True:
        img = pipe.recv()
        angle = predict(predictor, img, z)
        # calculate true angle
        true_angle = int(angle * 1600 + 700)
        print("sending angle:", true_angle)
        pipe.send(true_angle)


### 线程区
def detectProcess(predictor,pipe):
    while True:
        img = pipe.recv()
        # detection
        tensor_img = img.resize((256, 256), Image.BILINEAR)  #######resize 256*256
        if tensor_img.mode != 'RGB':
            tensor_img = tensor_img.convert('RGB')
        tensor_img = np.array(tensor_img).astype('float32')  # .transpose((2, 0, 1))  # HWC to CHW
        tensor_img -= 127.5
        tensor_img *= 0.007843
        tensor_img = tensor_img[np.newaxis, :]

        tensor = pm.PaddleTensor()
        tensor.dtype = pm.PaddleDType.FLOAT32
        tensor.shape = (1, 3, 256, 256)
        tensor.data = pm.PaddleBuf(tensor_img)
        paddle_data_feeds1 = [tensor]
        outputs1 = predictor.Run(paddle_data_feeds1)
        assert len(outputs1) == 1, 'error numbers of tensor returned from Predictor.Run function !!!'
        bboxes = np.array(outputs1[0], copy=False)
        print("sending bboxes")
        pipe.send(bboxes)


def videoProcess(pipe):
    out = cv2.VideoWriter('save.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (320, 240))
    try:
        while True:
            (frame, labels, scores, boxes) = pipe.recv()
            out.write(process_frame(frame,labels,scores,boxes))
    finally:
        out.release()
        print("Video Process Ended.")


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
    carline_predictor = load_model()

    '''##########################################################object  detect##########################################################'''
    init_train_parameters()
    detect_predictor = load_model_detect()

    vel = int(vels)
    lib_path = path + "/lib" + "/libart_driver.so"
    so = cdll.LoadLibrary
    lib = so(lib_path)
    car = "/dev/ttyUSB0"

    if (lib.art_racecar_init(38400, car.encode("utf-8")) < 0):
        raise
        pass
    # try:
    # start speed
    print("sending speed value:", vels)
    lib.send_cmd(1500, 0)
    lib.send_cmd(0, vels)
    # reset ZEBRA and RED and GREEN
    ZEBRA_SIGN = False
    RED_SIGN = False
    GREEN_SIGN = False
    # most possible status

    ### DEFINITION END
    NO_LIMIT = 0
    LIMIT = 1
    PARK = 2
    RED = 3
    GREEN = 4
    ZEBRA = 5
    BLOCK = 6
    #### DEFINITION
    # start multiprocessing
    (car_pipe1,car_pipe2) = multiprocessing.Pipe()
    (detect_pipe1, detect_pipe2) = multiprocessing.Pipe()
    (video_pipe1, video_pipe2) = multiprocessing.Pipe()
    p1 = multiprocessing.Process(target= carLineProcess,name="carline",args=(carline_predictor,car_pipe2),daemon=True)
    p2 = multiprocessing.Process(target= detectProcess,name="detect",args=(detect_predictor,detect_pipe2),daemon=True)
    p3 = multiprocessing.Process(target=videoProcess, name="video", args=(video_pipe2,),daemon=True)
    p1.start()
    p2.start()
    p3.start()
    try:
        while 1:
            nowtime2 = datetime.now()
            yuzhi_0 = 0
            yuzhi_3 = 0
            yuzhi_4 = 0
            ### SIGN DEFINITION START
            # judge zebra line
            zebra_line_detected = False
            # labels in no_limit(0),limit(1),park(2),red(3),green(4),zebra(5),block(6)
            while 1:
                nowtime = time.time()
                origin, img, frame = dataset(video)
                # 发送图片
                car_pipe1.send(img)
                detect_pipe1.send(origin)
                # 接收数据
                # calculate true angle
                true_angle = car_pipe1.recv()
                bboxes = detect_pipe1.recv()
                # tensor_img,img= read_image()  #  resize image
                t_labels = []
                t_scores = []
                t_boxes = []
                center_x = []
                center_y = []

                final_label = NO_LIMIT
                final_score = 0.0

                if len(bboxes.shape) == 1:
                    print("No object found in video")
                    STATE_value = False
                else:
                    STATE_value = False
                    labels = bboxes[:, 0].astype('int32') % classes
                    # scores -> angle , true_angle = 500 + 2000*scores
                    scores = bboxes[:, 1].astype('float32')
                    # pixels position
                    boxes = bboxes[:, 2:].astype('float32')
                    print("labels:", str(labels))
                    print("scores:", str(scores))
                    print("boxes:", str(boxes))
                    video_pipe1.send((frame,labels,scores,boxes))
                    ### start algorithm
                    # <- final_label
                    for i in range(len(labels)):
                        if scores[i] >= recog_rate:
                            # t_labels.append(label_dict[labels[i]])
                            # read sign
                            if labels[i] == ZEBRA:
                                ZEBRA_SIGN = True
                            elif labels[i] == RED:
                                RED_SIGN = True
                            elif labels[i] == GREEN:
                                GREEN_SIGN = True
                            # else:
                            #     # process
                            #     if scores[i] > final_score:
                            #         final_label = labels[i]
                            #         final_score = scores[i]
                            #     # collect infos, of no use currently
                            #     t_labels.append(labels[i])
                            #     print("t_labels:", str(t_labels))
                            #     t_scores.append(scores[i])
                            #     print("t_scores:", str(t_scores))
                            #     center_x.append(int((boxes[i][0] + boxes[i][2]) / 2))
                            #     center_y.append((boxes[i][1] + boxes[i][3]) / 2)
                            #     print("the center coordinate of object:", center_x, "   ", center_y)
                    STATE_value = True

                ################################################################################################
                ### preprocessing data (data、sign)
                # TODO BLOCK Judgement
                # TODO

                # Judge SIGN
                if RED_SIGN == True and ZEBRA_SIGN == True:
                    # V2: only recognize both SIGNs
                    final_label = PARK
                if GREEN_SIGN == True:
                    final_label = NO_LIMIT
                ### start sending proccessed command
                print("sending: angle: %d, label: %d" % (true_angle, final_label))
                # user_cmd(STATE_value,t_labels,t_scores,center_x,center_y,vel,a)
                lib.send_cmd(true_angle, int(final_label))
                # if final_label == 2:
                #    input("stop")
                # if final_label == 1:
                #    input("limit")
                print("the time of predict:", time.time() - nowtime)
    except KeyboardInterrupt as e:
        print("keyboard detected")
        lib.send_cmd(1500, 2)
        # stop car
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p1.join()
        p2.join()
        p3.join()
        exit(0)
    finally:
        detect_pipe1.close()
        detect_pipe2.close()
        car_pipe1.close()
        car_pipe2.close()
        video_pipe1.close()
        video_pipe2.close()
        print('exit')

"""
2020-07-16 16:59:28
DEBUG INFO:
114
the time of predict: 0.052869319915771484
Before func_expand image_shape: (128, 128, 3)
vedio image_shape: (1, 128, 128, 3)
predict img.shape: (1, 128, 128, 3)
predict z.shape: (1, 128, 128, 3)
[ 0.65283203]
outputs1 value: [<paddlemobile.PaddleTensor object at 0x7f898c3ce0>]
bboxes.shape (2, 6)
labels: [1 2]
scores: [ 0.58105469  0.125     ]
boxes: [[   5.2265625    27.375       520.5         607.        ]
 [  35.90625       7.70703125  519.          556.        ]]
t_labels: [1]
t_scores: [0.58105469]
the center coordinate of object: [262]     [317.1875]
t_labels: [1, 2]
t_scores: [0.58105469, 0.125]
the center coordinate of object: [262, 277]     [317.1875, 281.853515625]
angle: 1591, throttle: 1535
******************************************angle: 1591, throttle: 1535
*******************ignore  detect
the send buff is:
00 aa ff 05 37 06 41 55



"""
