'''
import cv2 
import numpy as np
import time
from PIL import Image

def load_image(cap):

    lower_hsv = np.array([156, 43, 46])
    upper_hsv = np.array([180, 255, 255])
    lower_hsv1 = np.array([0, 43, 46])
    upper_hsv1 = np.array([10, 255, 255])
    # ref, frame = cap.read()
    frame = cap
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

img = cv2.imread("C:/Users/Lenovo/Desktop/img/test.jpg")



load_image(img)
cv2.namedWindow("im1", cv2.WINDOW_NORMAL)   # namedWindow 窗口名称+窗口类型 cv2.WINDOW_NORMAL（窗口大小可以拖动）
cv2.imshow("im1", img)
cv2.waitKey(0)
cv2.destroyWindow("im1")
'''
def cnn_model(image):
    conv1 = fluid.layers.conv2d(input=image, num_filters=32, filter_size=5, stride=2, act='relu')
    conv2 = fluid.layers.conv2d(input=conv1, num_filters=32, filter_size=5, stride=2)
    bn0 = fluid.layers.batch_norm(input=conv2,act='relu')
    conv3 = fluid.layers.conv2d(input=bn0, num_filters=64, filter_size=5, stride=2, act='relu')
    conv4 = fluid.layers.conv2d(input=conv3, num_filters=64, filter_size=3, stride=2)
    bn1 = fluid.layers.batch_norm(input=conv4,act='relu')
    conv5 = fluid.layers.conv2d(input=bn1, num_filters=128, filter_size=3, stride=1)
    bn2 = fluid.layers.batch_norm(input=conv5,act='relu')
    fc1 = fluid.layers.fc(input=bn2, size=128, act=None)
    fc2 = fluid.layers.fc(input=fc1, size=64, act=None)
    predict = fluid.layers.fc(input=fc2, size=1)
    return predict


