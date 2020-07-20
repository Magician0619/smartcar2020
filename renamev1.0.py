'''
@Author: Magician
@Date: 2020-07-20 22:54:03
@LastEditors: HK
@LastEditTime: 2020-07-20 23:08:38
@Description: file content
@FilePath: \smartcar2020\renamev1.0.py
'''
# -*- coding:utf8 -*-

'''
@Author: Magician
@Date: 2020-07-17 21:46:56
@LastEditors: HK
@LastEditTime: 2020-07-18 09:42:29
@Description: 批量重命名图片
'''




import os
import re


#E:\智能车\人工智能组资料\人工智能创意赛复赛\数据集\赛道线\车道线20圈下午\data\img
src_path = "E:/智能车/人工智能组资料/人工智能创意赛复赛/数据集/赛道线/车道线0720中午 (3)/data/img"
dst_path = "E:/智能车/人工智能组资料/人工智能创意赛复赛/数据集/赛道线/车道线20圈下午/data/img"

filelist = os.listdir(src_path)   #获取文件路径
total_num = len(filelist)  #获取文件长度（个数）
i = 12205  #表示文件的命名是从1开始的
for j in range(8043):  #多少个项目
    item = str(j) +'.jpg'
    it = str(i+j) +'.jpg'
    if item in filelist:
        
        src = os.path.join(os.path.abspath(src_path), item)
        dst = os.path.join(os.path.abspath(dst_path),  it)#处理后的格式也为jpg格式的，当然这里可以改成png格式
        # 这种情况下的命名格式为xn000.jpg形式，可以自主定义想要的格式

        os.rename(src, dst)
     