#代码参考b站python+opencv3.3视频教学 基础入门 贾志刚
import cv2 as cv
import numpy as np


def access_pixels(image):
    print(image.shape)
    #shape返回的是图像的行数，列数，色彩通道数。
	#行数其实对应于坐标轴上的y,即表示的是图像的高度
	#列数对应于坐标轴上的x，即表示的是图像的宽度
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width : %s, height :%s, channels: %s" % (width,height,channels))
#循环每个像素
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv=image[row,col,c]
                image[row,col,c]=255-pv
    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("pixels_demo",image)


def create_image():
	#创建400*400，3通道的全0矩阵
    img=np.zeros([400,400,3],np.uint8)
    #opencv 加载图像是BGR模式：第三个参数0为蓝色，1为绿色，2为红色
    img[:,:,0]=np.ones([400,400])*255 #显示为全蓝色的400*400的图像
    cv.imshow("new image",img)

    # img=np.ones([400,400,1],np.uint8)
    # img=img*0
    # cv.imshow("new image",img)
    #cv.imwrite("D:/Documents/Pictures/exam.jpg",img)


src = cv.imread("C:/Users/Lenovo/Desktop/img/test.jpg")
print(src.shape)
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
t1=cv.getTickCount()
#access_pixels(src)
create_image()
t2=cv.getTickCount()
time=(t2-t1)/cv.getTickFrequency()
print("time: %s ms"%(time*1000))
cv.waitKey(0)

cv.destroyAllWindows()
