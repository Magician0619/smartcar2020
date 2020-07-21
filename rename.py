# -*- coding:utf8 -*-

'''
#@Author: Magician
#@Date: 
#@Description: file rename in correct order

Copyright 2020 by Magician
'''

import os
import re

#将src中的文件批处理命名放到dst中
#C:\Users\Lenovo\Desktop\0721晚上赛道线-ljc\0721晚上赛道线-ljc\1\data\img
src_path = "C:/Users/Lenovo/Desktop/0721晚上赛道线-ljc/0721晚上赛道线-ljc/5/data/img"
dst_path = "C:/Users/Lenovo/Desktop/0721晚上赛道线-ljc/0721晚上赛道线-ljc/6/data/img"

srclist = os.listdir(src_path)   #获取文件路径
src_num = len(srclist)  #获取文件长度（个数）

dstlist = os.listdir(dst_path)   #获取文件路径
dst_num = len(dstlist)  #获取文件长度（个数）

#src文件夹中文件最好要比dst中的要少，不要引起重名的情况！！！

def img_rename( ):

    for j in range(src_num):
        src = str(j) +'.jpg'
        dst = str(dst_num+j) +'.jpg'
        if src in srclist:
            
            src = os.path.join(os.path.abspath(src_path), src)
            dst = os.path.join(os.path.abspath(dst_path),  dst)
            
            os.rename(src, dst)

if __name__=='__main__':
    img_rename()