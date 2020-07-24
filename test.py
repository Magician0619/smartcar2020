import cv2 
import numpy as np

img = cv2.imread("C:/Users/Lenovo/Desktop/img/test.jpg")

print(img.shape)
print(img)
cv2.namedWindow("im1", cv2.WINDOW_NORMAL)   # namedWindow 窗口名称+窗口类型 cv2.WINDOW_NORMAL（窗口大小可以拖动）
cv2.imshow("im1", img)
cv2.waitKey(0)
cv2.destroyWindow("im1")
