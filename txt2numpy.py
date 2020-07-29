# -*- coding:utf-8 -*- 

'''
#@Author: Magician
#@Date: 
#@Description: 

Copyright 2020 by Magician
'''

import numpy as np
import os

path = "C:/Users/Lenovo/Desktop/07252628_carline/072526_carline/data"


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
