# -*- coding:utf-8 -*- 

'''
#@Author: Magician
#@Date: 
#@Description: 

Copyright 2020 by Magician
'''

import numpy as np
import os

path = "C:/Users/Lenovo/Desktop/2020年7月29日1649特殊情况直角/1"


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

if __name__ == '__main__':
    txt_2_numpy()
