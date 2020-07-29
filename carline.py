#created by BOXIAO  27/09/2017
#all rights reserved 
import os
import cv2
import sys
import numpy as np
import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import feature 


#预处理图像膨胀和腐蚀（可选）
def preprocessimage(img):
    kernel=np.uint8(np.zeros((5,5)))
    for x in range(5):
        kernel[x,2]=1
        kernel[2,x]=1
    eroded=cv2.erode(img,kernel)
    dilated = cv2.dilate(img, kernel)
    result = cv2.absdiff(dilated, eroded)
    return result
#scale所得的车道线，使其充满屏幕
def scale_lines(chooses_lines):
    final_lines=[]
    for line in choose_lines:
        p0, p1 = line
        k = np.float32(((p0[1]-p1[1])*1.0)/(1.0*(p0[0]-p1[0])))
        b = np.float32(p1[1] - k*p1[0])
        y1=0
        x1=-b/k
        if k < 0:
            x2=0
            y2=b
        else:
            x2=640
            y2=640*k+b
        line=((x1,y1),(x2,y2))
        final_lines.append(line)
    return final_lines


if __name__ == '__main__':
    input_dir=sys.argv[1]
    output_dir=sys.argv[2]
    data_dir='C:/Users/Lenovo/Desktop/2020-07-26 230802/data1/img'
    frames_list = os.listdir(input_dir)
    frames_list.sort()
    for f, frame in enumerate(frames_list):
        img = cv2.imread(os.path.join(data_dir,input_dir,frame))
        ROI=[330,480,0,640]
        ROI_img=img[ROI[0]:ROI[1],ROI[2]:ROI[3]]
        image = cv2.cvtColor(ROI_img,cv2.COLOR_BGR2GRAY)
        edges = feature.canny(image, sigma=2, low_threshold=2, high_threshold=25)
        all_lines = st.probabilistic_hough_line(edges, threshold=10, line_length=70,line_gap=0)
        #根据斜率将直线分为两组：左直线和右直线
        left_lines = []
        right_lines = []
        for line in all_lines:
            p0, p1 = line
            theta = np.abs(np.arctan2((p0[0] - p1[0]), (p0[1] - p1[1])))
            if theta > 1:
                left_lines.append(line)
            else:
                right_lines.append(line)
        choose_lines = [left_lines[0], right_lines[0]]
       # scaling left and right lines
        lines=scale_lines(choose_lines)
        #绘图并且保存
        fig, (ax0,ax1,ax2,ax3) = plt.subplots(1, 4, figsize=(16, 6))
        plt.tight_layout()
        ax0.imshow(img)
        ax0.set_title('Input image')
        ax0.set_axis_off()
        ax1.imshow(edges,plt.cm.gray)
        ax1.set_title('Canny deges')
        ax1.set_axis_off()
        ax2.imshow(edges*0)
        for line in all_lines:
            p0, p1 = line
            ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))
        row2, col2 = image.shape
        ax2.axis((0, col2, row2, 0))
        ax2.set_title('Probabilistic Hough')
        ax2.set_axis_off()
        ax3.imshow(image, plt.cm.gray)
        for line in lines:
            p0, p1 = line
            ax3.plot((p0[0], p1[0]), (p0[1], p1[1]),linewidth=10.0)
        ax3.set_title('Output image')
        ax3.set_axis_off()
        plt.savefig(os.path.join(data_dir,output_dir,frame))
        plt.show()
