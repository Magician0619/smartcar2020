# -*- coding:utf8 -*-

'''
@Author: Magician
@Date: 2020-07-17 21:46:56
@LastEditors: HK
@LastEditTime: 2020-07-20 22:31:58
@Description: 批量重命名图片
'''


import os
import re

class BatchRename():
    '''
    批量重命名文件夹中的图片文件

    '''
    def __init__(self):
        # self.path = "E:/智能车/人工智能组资料/人工智能创意赛复赛/数据集/赛道线/车道线0717未加红绿灯/data/img" #表示需要命名处理的文件夹
        self.path ="E:/智能车/人工智能组资料/人工智能创意赛复赛/数据集/赛道线/车道线0720中午 (2)/data/img"

    def rename(self):
        filelist = os.listdir(self.path)   #获取文件路径
        total_num = len(filelist)  #获取文件长度（个数）
        i = 12205  #表示文件的命名是从12921开始的
        for item in filelist:
            # for j in os.listdir(r"./file"):               
            if item.endswith('.jpg'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path),  format(str(i)) + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                # 这种情况下的命名格式为xn000.jpg形式，可以自主定义想要的格式
                try:
                    os.rename(src, dst)
                    #print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        #print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()