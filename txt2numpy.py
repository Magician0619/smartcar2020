# -*- coding:utf-8 -*- 

'''
#@Author: Magician
#@Date: 
#@Description: 

Copyright 2020 by Magician
'''

import numpy as np
import os

path = "D:/2020年8月16日184321-1560（2500）overtake（需要改hsv和重写numpy）"



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
