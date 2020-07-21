'''
@Author: Magician
@Date: 2020-07-21 10:39:32
@LastEditors: HK
@LastEditTime: 2020-07-21 10:50:10
@Description: file content
@FilePath: \smartcar2020\mask.py
'''
import  cv2
import numpy as np


lower_hsv = np.array([25, 75, 190])
upper_hsv = np.array([40, 255, 255])

img = cv2.imread("C:/Users/Lenovo/Desktop/fsdownload/data/img/761.jpg")
frame = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask0 = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)

mask = mask0 #+ mask1


img = Image.fromarray(mask)
cv2.imwrite('test1.jpg', img)
img = img.resize((128, 128), Image.ANTIALIAS)
#img = cv2.resize(img, (128, 128))
img = np.array(img).astype(np.float32)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = img / 255.0;
img = np.expand_dims(img, axis=0)
print("image____shape:",img.shape)
