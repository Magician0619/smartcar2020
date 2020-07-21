'''
@Author: Magician
@Date: 2020-07-20 16:13:45
@LastEditors: HK
@LastEditTime: 2020-07-20 16:16:15
@Description: file content
@FilePath: \smartcar2020\src\txt2numpy.py
'''
path = "C:/Users/Lenovo/Desktop/0721晚上赛道线-ljc/0721晚上赛道线-ljc/6/data" 

import numpy as np

def txt_2_numpy():
    angledata = []
    data = []
    file = open(path+"/data.txt","r")
    for line in file.readlines():
        line = line.strip('\n')
        angledata.append(int(line))
    angle = np.array(angledata)
    np.save(path+"/data.npy", angle,False)
    file.close()

txt_2_numpy()