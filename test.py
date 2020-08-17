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

def _start_worker(self):
    try:
        while True:
            time.sleep(0.01)
            self.request_dict_lock.acquire()
            for key in list(self.request_dict):
                if self.request_dict[key][-1] <= time.time():
                    self._driver.send_cmd(self.request_dict[key][0],self.request_dict[key][1])
                    if self.request_dict[key][0] == 0:
                        print("SPEED:",self.request_dict[key])                   
                    self.request_dict.pop(key)
            self.request_dict_lock.release()
    except Exception as e:
        print(str(e))
        return 

elif axis == "hat0y":
    # up -32767, down 32767
    fvalue = value / 32767
    data[0] = int(data[0] - 5*fvalue)
    speed_car = data[0]
    print("new speed:",data[0])
    axis_states[axis] = fvalue
    lib.send_cmd(data[0],angle_car)









